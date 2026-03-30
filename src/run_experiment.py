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

from utils.config import Config, parse_args
from utils.readers import load_mysun_dataset
from utils.model_parser import get_model
from utils.opt_parser import get_optimizer
from utils.scheduler_parser import get_scheduler
from utils.datasets import sun_depth_dataset
from train_FM import train, evaluate
from utils.losses import FlowMatchingLoss
from visualize_final import run_distributed_visualizations

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
        targets = ('ResidualBlock', 'ResidualCrossAttentionBlock', 'Conv2d', 'ConvTranspose2d')
        
        for name, sub_module in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            if sub_module.__class__.__name__ in targets:
                def hook_fn(m, i, o, name=full_name):
                    shape = list(o.shape) if torch.is_tensor(o) else "Multi-tensor"
                    print(f"{name:<50} | {shape}")
                hooks.append(sub_module.register_forward_hook(hook_fn))
            
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
    torch.backends.cudnn.benchmark = True
    if device.type == 'cuda':
        print(f"Using GPU {local_rank}")
        torch.cuda.manual_seed_all(Config.RANDOM_SEED)

    print(f"Rank {local_rank}: Loading dataset paths...")
    x_paths, y_paths = load_mysun_dataset(Config.MYSUNRGBD_DATA_DIR)

    train_transform = v2.Compose([
        v2.RandomResizedCrop(size=(Config.data_config['side_pixels'], Config.data_config['side_pixels']), scale=(0.16, 1.0), ratio=(1.0, 1.0), antialias=True),
        v2.RandomHorizontalFlip(p=0.5)
    ])
    val_transform = v2.Compose([
        v2.CenterCrop(Config.data_config['side_pixels'])
    ])

    x_train, x_test, y_train, y_test = train_test_split(x_paths, y_paths, test_size=Config.data_config['val_split'], random_state=Config.RANDOM_SEED)
    x_train, x_val, y_train, y_val   = train_test_split(x_train, y_train, test_size=Config.data_config['val_split'], random_state=Config.RANDOM_SEED)

    train_dataset = sun_depth_dataset(x_train, y_train, cache_size=Config.data_config['cache_size'], transform=train_transform)
    val_dataset   = sun_depth_dataset(x_val,   y_val,   cache_size=Config.data_config['cache_size'], transform=val_transform)
    test_dataset  = sun_depth_dataset(x_test,  y_test,  cache_size=Config.data_config['cache_size'], transform=val_transform)

    train_sampler = DistributedSampler(train_dataset)              if is_distributed else None
    val_sampler   = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None

    train_loader = DataLoader(train_dataset, batch_size=Config.data_config['batch_size'], shuffle=(train_sampler is None), sampler=train_sampler,
                              num_workers=Config.data_config['num_workers_train'], pin_memory=True, persistent_workers=True, prefetch_factor=2, drop_last=True)
    val_loader   = DataLoader(val_dataset,  batch_size=Config.data_config['batch_size'], num_workers=Config.data_config['num_workers'], shuffle=False, sampler=val_sampler, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=Config.data_config['batch_size'], num_workers=Config.data_config['num_workers'], shuffle=False, pin_memory=True)

    # Resuming logic
    resume_path = "checkpoint_latest.pth"
    final_path  = "model_final.pth"
    start_epoch = 1
    best_val    = float('inf')
    ema_state   = None
    ckpt        = None
    loaded_from_final = False
    load_path = None

    if is_main_process:
        if os.path.exists(final_path):
            load_path = final_path
            loaded_from_final = True
            print(f"\n[Recovery] Found {final_path}. Preparing for visualizations...")
        elif os.path.exists(resume_path):
            load_path = resume_path
            print(f"\n[Reconciliation] Checkpoint found at {resume_path}...")

    if is_distributed:
        info_list = [load_path, loaded_from_final]
        dist.broadcast_object_list(info_list, src=0)
        load_path, loaded_from_final = info_list[0], info_list[1]

    if load_path:
        ckpt_raw = torch.load(load_path, map_location='cpu', weights_only=False)
        
        if loaded_from_final and ('model_state_dict' not in ckpt_raw):
            if is_main_process:
                print("Legacy model_final.pth detected (raw weights only). Bypassing config check.")
            ckpt = {'model_state_dict': ckpt_raw, 'mlflow_run_id': None}
            start_epoch = Config.training_config['epochs'] + 1 
        else:
            ckpt = ckpt_raw
            if 'config' in ckpt:
                saved_model_cfg = ckpt['config'].get('model_config', {})
                mismatches = []
                for key, saved_val in saved_model_cfg.items():
                    current_val = Config.model_config.get(key)
                    if current_val != saved_val:
                        mismatches.append(f"  - {key}: {current_val} -> {saved_val} (Forced)")
                        Config.model_config[key] = saved_val
                
                if mismatches and is_main_process:
                    print("WARNING: Overriding current Config.model_config with Checkpoint values:")
                    print("\n".join(mismatches))
                    print("If you intended to use the NEW config, please rename/delete the checkpoint.\n")

    if is_distributed:
        config_list = [Config.model_config]
        dist.broadcast_object_list(config_list, src=0)
        Config.model_config = config_list[0]

    model = get_model(Config.model_config).to(device)
    model = model.to(memory_format=torch.channels_last)

    optimizer = get_optimizer(model, Config.training_config)
    scheduler = get_scheduler(optimizer, Config.training_config, steps_per_epoch=len(train_loader))

    if ckpt is not None:
        if is_main_process:
            print(f"Loading weights from {load_path}...")
        
        model.load_state_dict(ckpt['model_state_dict'])
        
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if scheduler and ckpt.get('scheduler_state_dict'):
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            
        if 'epoch' in ckpt:
            start_epoch = ckpt['epoch'] + 1
        if 'best_val' in ckpt:
            best_val = ckpt['best_val']
        if 'ema_state_dict' in ckpt:
            ema_state = ckpt['ema_state_dict']
            
        if loaded_from_final:
            start_epoch = Config.training_config['epochs'] + 1



    if is_main_process:
        debug_model_architecture(
            model,
            input_size=(1, 1, Config.data_config['side_pixels'], Config.data_config['side_pixels']),
            cond_size=(1, 3, Config.data_config['side_pixels'], Config.data_config['side_pixels'])
        )

    if is_distributed:
        dist.barrier()
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, static_graph=True)

    if is_main_process:
        mlflow.set_tracking_uri("file://" + Config.MLFLOW_DIR)
        mlflow.set_experiment(Config.experiment_name)
        run_id = ckpt.get('mlflow_run_id') if (ckpt is not None) else None
        if run_id:
            print(f"Resuming MLflow run: {run_id}")
            mlflow.start_run(run_id=run_id)
        else:
            mlflow.start_run(run_name=Config.run_name)
            mlflow.log_params(Config.model_config)
            mlflow.log_params(Config.training_config)
            mlflow.set_tag("problem", "diffusion_nyu")

    if is_distributed:
        dist.barrier()

    ema_state_dict = train(
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
        edge_weight=Config.training_config.get('edge_weight', 0.0),
        x1_weight=Config.training_config.get('x1_weight', 0.0),
        weight_type=Config.training_config.get('weight_type', 'none'),
        time_sampling=Config.training_config.get('time_sampling', 'uniform'),
        side_pixels=Config.data_config['side_pixels'],
        patience=Config.training_config['patience'],
        ema_decay=Config.training_config.get('ema_decay', 0.999),
        cond_drop_prob=Config.training_config.get('cond_drop_prob', 0.20),
        is_distributed=is_distributed,
        train_sampler=train_sampler,
        is_main_process=is_main_process,
        log_depth=Config.training_config.get('log_depth', False),
        start_epoch=start_epoch,
        best_val=best_val,
        ema_state=ema_state
    )

    m = model.module if hasattr(model, 'module') else model

    if ema_state_dict is not None:
        if is_main_process:
            print("\nApplying EMA weights for final Test Evaluation and Visualizations...")
        cleaned_ema = {
            k.replace('module.', ''): v
            for k, v in ema_state_dict.items()
            if k.replace('module.', '') != 'n_averaged'
        }
        m.load_state_dict(cleaned_ema)

    if is_main_process and not loaded_from_final:
        loss_fn = FlowMatchingLoss(
            base_type=Config.training_config['loss'],
            grad_weight=Config.training_config.get('grad_weight', 0.5),
            si_weight=Config.training_config.get('si_weight', 0.0),
            edge_weight=Config.training_config.get('edge_weight', 0.0),
            x1_weight=Config.training_config.get('x1_weight', 0.0),
            log_depth=Config.training_config.get('log_depth', False)
        )
        print("Running final evaluation on Test Set...")
        test_metrics_raw = evaluate(
            model=m,
            loss_fn=loss_fn,
            dataloader_val=test_loader,
            device=device,
            weight_type=Config.training_config.get('weight_type', 'none'),
            time_sampling=Config.training_config.get('time_sampling', 'uniform')
        )
        test_metrics = {k.replace('val_', 'test_'): v for k, v in test_metrics_raw.items()}
        print(f"Test Loss (Total): {test_metrics['test_total']:.6f}")
        print(f"Test Std L1 (t=0.5): {test_metrics.get('test_l1_standard', 0.0):.6f}")
        mlflow.log_metrics(test_metrics, step=Config.training_config['epochs'])
        final_ckpt = {
            'epoch': Config.training_config['epochs'],
            'model_state_dict': m.state_dict(),
            'config': {
                'model_config': Config.model_config,
                'training_config': Config.training_config,
                'data_config': Config.data_config
            },
            'mlflow_run_id': mlflow.active_run().info.run_id if mlflow.active_run() else None
        }
        torch.save(final_ckpt, "model_final.pth")
        mlflow.log_artifact("model_final.pth")
        
        if os.path.exists("checkpoint_latest.pth"):
            os.remove("checkpoint_latest.pth")
            print("Deleted intermediate checkpoint 'checkpoint_latest.pth' to ensure a fresh start next run.")

    if is_distributed:
        dist.barrier()
        world_size = dist.get_world_size()
    else:
        world_size = 1

    if is_main_process:
        print("Initializing full-resolution test dataset for visualizations...")
        
    test_transform_full = v2.Identity()
    test_dataset_vis = sun_depth_dataset(
        x_test, 
        y_test, 
        cache_size=Config.data_config['cache_size'],
        transform=test_transform_full
    )

    run_distributed_visualizations(
        model=model, 
        test_dataset=test_dataset_vis, 
        device=device, 
        local_rank=local_rank, 
        world_size=world_size, 
        is_main_process=is_main_process
    )

    if is_main_process:
        if os.path.exists(final_path):
            archive_name = "model_final_plotted.pth"
            os.rename(final_path, archive_name)
            print(f"Renamed '{final_path}' to '{archive_name}' to ensure a fresh start next run.")
            
        mlflow.end_run()

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()
    
if __name__ == "__main__":
    parse_args()
    run()