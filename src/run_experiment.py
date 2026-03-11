import os

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mlflow
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import v2


from utils.config import Config
from utils.readers import load_mnist_images, load_mnist_labels, load_nyu_labeled_subset, load_sun_rgbd_subset
from utils.model_parser import get_model
from utils.opt_parser import get_optimizer
from utils.scheduler_parser import get_scheduler
from utils.datasets import mnist_dataset, nyu_depth_dataset, sun_depth_dataset
from utils.visualize import create_flow_animation, create_depth_flow_animation
from train_FM import train, evaluate
from evolve import save_flow_evolution, save_full_flow_evolution
from utils.losses import FlowMatchingLoss

def debug_model_architecture(model, input_size=(1, 1, 128, 128), cond_size=(1, 3, 128, 128)):
    print("\n" + "="*60)
    print("DETAILED MULTI-SCALE ARCHITECTURE REPORT")
    print("="*60)
    
    device = next(model.parameters()).device
    dummy_x = torch.randn(*input_size).to(device)
    dummy_t = torch.rand(input_size[0]).to(device)
    dummy_y = torch.randn(*cond_size).to(device)
    
    print(f"{'Layer / Sub-module':<50} | {'Output Shape'}")
    print("-" * 65)

    hooks = []
    def register_recursive_hooks(module, prefix=""):
        # We target specific blocks to avoid printing every single GroupNorm/GELU
        # This will capture your Stages, Residuals, and Cross-Attentions
        targets = ('ResidualBlock', 'ResidualCrossAttentionBlock', 'Conv2d', 'ConvTranspose2d')
        
        for name, sub_module in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            if sub_module.__class__.__name__ in targets:
                def hook_fn(m, i, o, name=full_name):
                    shape = list(o.shape) if torch.is_tensor(o) else "Multi-tensor"
                    print(f"{name:<50} | {shape}")
                hooks.append(sub_module.register_forward_hook(hook_fn))
            
            # Recurse deeper into ModuleLists and ModuleDicts
            register_recursive_hooks(sub_module, full_name)

    register_recursive_hooks(model)

    try:
        with torch.no_grad():
            model(dummy_x, dummy_t, dummy_y)
    finally:
        for h in hooks:
            h.remove()
    
    total_params = sum(p.numel() for p in model.parameters())
    print("-" * 65)
    print(f"Total Parameters: {total_params:,}")
    print("="*60 + "\n")

def run():
    is_distributed = "LOCAL_RANK" in os.environ
    if is_distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)
    else:
        local_rank = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    is_main_process = local_rank == 0

    torch.manual_seed(Config.RANDOM_SEED)
    if device.type == 'cuda':
        print(f"Using GPU {local_rank}")
        torch.cuda.manual_seed_all(Config.RANDOM_SEED)


    cache_file = "dataset_cache.pt"

    if is_main_process:
        print(f"Rank {local_rank}: Reading MAT file and processing datasets...")
        #x, y = load_nyu_labeled_subset(os.path.join(Config.NYU_DATA_DIR, 'nyu_depth_v2_labeled.mat'))
        x, y = load_sun_rgbd_subset(Config.SUNRGBD_DATA_DIR)

        train_transform = v2.Compose([
            v2.RandomResizedCrop(size=(Config.data_config['side_pixels'], Config.data_config['side_pixels']), scale=(0.16, 1.0), ratio=(1.0, 1.0), antialias=True),
            v2.RandomHorizontalFlip(p=0.5)
        ])
        
        val_transform = v2.Compose([
            v2.CenterCrop(Config.data_config['side_pixels'])
        ])

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=Config.data_config['val_split'], random_state=Config.RANDOM_SEED)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=Config.data_config['val_split'], random_state=Config.RANDOM_SEED)

        train_dataset = sun_depth_dataset(x_train, y_train, cache_size=Config.data_config['cache_size'], transform=train_transform)
        val_dataset  = sun_depth_dataset(x_val, y_val, cache_size=Config.data_config['cache_size'], transform=val_transform)
        test_dataset = sun_depth_dataset(x_test, y_test, cache_size=Config.data_config['cache_size'], transform=val_transform)
        
        torch.save({'train': train_dataset, 'val': val_dataset, 'test': test_dataset}, cache_file)
        print(f"Rank {local_rank}: Dataset cached to {cache_file}")


    if is_distributed:
            dist.barrier()

    if not is_main_process:
        print(f"Rank {local_rank}: Loading pre-processed dataset directly from cache...")
        cached_data = torch.load(cache_file, weights_only=False)
        train_dataset = cached_data['train']
        val_dataset   = cached_data['val']
        test_dataset  = cached_data['test']

    train_sampler = DistributedSampler(train_dataset) if is_distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None

    train_loader = DataLoader(train_dataset, batch_size=Config.data_config['batch_size'], shuffle=(train_sampler is None), sampler=train_sampler,
                              num_workers=Config.data_config['num_workers_train'], pin_memory=True, persistent_workers=True, prefetch_factor=2, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=Config.data_config['batch_size'], num_workers=Config.data_config['num_workers'], shuffle=False, sampler=val_sampler, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=Config.data_config['batch_size'], num_workers=Config.data_config['num_workers'], shuffle=False, pin_memory=True)

    model = get_model(Config.model_config).to(device)
    
    if is_main_process:
            debug_model_architecture(
                model, 
                input_size=(1, 1, Config.data_config['side_pixels'], Config.data_config['side_pixels']),
                cond_size=(1, 3, Config.data_config['side_pixels'], Config.data_config['side_pixels'])
            )
            
    if is_distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = get_optimizer(model, Config.training_config)
    scheduler = get_scheduler(optimizer, Config.training_config, steps_per_epoch=len(train_loader))

    if is_main_process:
        mlflow.set_tracking_uri("file://" + Config.MLFLOW_DIR)
        mlflow.set_experiment(Config.experiment_name)
        run_obj = mlflow.start_run(run_name=Config.run_name)
        mlflow.log_params(Config.model_config)
        mlflow.log_params(Config.training_config)
        mlflow.set_tag("problem", "diffusion_nyu")

    if is_distributed:
        dist.barrier()

    train(
        model=model,
        optimizer=optimizer,
        epochs=Config.training_config['epochs'],
        scheduler=scheduler,
        dataloader_train=train_loader,
        dataloader_val=val_loader,
        device=device,
        loss_type=Config.training_config['loss'],
        grad_weight=Config.training_config.get('grad_weight', 0.5),
        si_weight=Config.training_config.get('si_weight', 0.0),
        weight_type=Config.training_config.get('weight_type', 'none'),
        time_sampling=Config.training_config.get('time_sampling', 'uniform'),
        side_pixels=Config.data_config['side_pixels'],
        patience=Config.training_config['patience'],
        ema_decay=Config.training_config.get('ema_decay', 0.999),
        cond_drop_prob=Config.training_config.get('cond_drop_prob', 0.20),
        is_distributed=is_distributed, 
        train_sampler=train_sampler,
        is_main_process=is_main_process
    )

    if is_main_process:
        loss_fn = FlowMatchingLoss(
            base_type=Config.training_config['loss'], 
            grad_weight=Config.training_config.get('grad_weight', 0.5), 
            si_weight=Config.training_config.get('si_weight', 0.0),
            edge_weight=Config.training_config.get('edge_weight', 0.0)
        )
        
        test_metrics = evaluate(model, loss_fn, test_loader, device, Config.training_config.get('weight_type', 'none'), Config.training_config.get('time_sampling', 'uniform'))
        print(f"Test Loss (Total): {test_metrics['val_total']:.6f}")
        mlflow.log_metric("test_loss", test_metrics['val_total'], step=Config.training_config['epochs'])

        m = model.module if hasattr(model, 'module') else model
        torch.save(m.state_dict(), "model_final.pth")
        mlflow.log_artifact("model_final.pth")
        
        print("Generating Full-Resolution Patch-Stitched Animations...")
        
        test_dataset.transform = None
        train_dataset.transform = None

        guidance_scales = Config.training_config.get('guidance_scale', [1.5])

        for ii in range(5):
            for gs in guidance_scales:
                gt_depth, rgb_label = test_dataset[ii] 
                
                snapshots = save_full_flow_evolution(
                    model=m, 
                    label=rgb_label, 
                    device=device, 
                    num_steps=100, 
                    patch_size=Config.data_config['side_pixels'], 
                    stride=64,     
                    guidance_scale=gs
                )
                
                torch.save(snapshots, f"snapshots_test_{ii}_{gs}.pt")
                mlflow.log_artifact(f"snapshots_test_{ii}_{gs}.pt")
                
                create_depth_flow_animation(
                    snapshots, 
                    filename=f"flow_evolution_test_log_{ii}_{gs}.gif", 
                    n_steps=100, 
                    timing_mode='logarithmic', 
                    gt_depth=gt_depth.unsqueeze(0) 
                )

        for ii in range(5):
            for gs in guidance_scales:
                gt_depth, rgb_label = train_dataset[ii] 
                
                snapshots = save_full_flow_evolution(
                    model=m, 
                    label=rgb_label, 
                    device=device, 
                    num_steps=100, 
                    patch_size=Config.data_config['side_pixels'], 
                    stride=64, 
                    guidance_scale=gs
                )
                
                torch.save(snapshots, f"snapshots_train_{ii}_{gs}.pt")
                mlflow.log_artifact(f"snapshots_train_{ii}_{gs}.pt")
                
                create_depth_flow_animation(
                    snapshots, 
                    filename=f"flow_evolution_train_log_{ii}_{gs}.gif", 
                    n_steps=100, 
                    timing_mode='logarithmic', 
                    gt_depth=gt_depth.unsqueeze(0)
                )

        mlflow.end_run()

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()
    
if __name__ == "__main__":
    run()