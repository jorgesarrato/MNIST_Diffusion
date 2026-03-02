import os

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mlflow

from utils.config import Config
from utils.readers import load_mnist_images, load_mnist_labels, load_nyu_labeled_subset
from utils.model_parser import get_model
from utils.opt_parser import get_optimizer
from utils.scheduler_parser import get_scheduler
from utils.datasets import mnist_dataset, nyu_depth_dataset
from utils.visualize import create_flow_animation, create_depth_flow_animation
from train_FM import train, evaluate
from evolve import save_flow_evolution

def run():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(Config.RANDOM_SEED)
    if device == 'cuda':
        torch.cuda.manual_seed_all(Config.RANDOM_SEED)

    x, y = load_nyu_labeled_subset(os.path.join(Config.NYU_DATA_DIR, 'nyu_depth_v2_labeled.mat'))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=Config.data_config['val_split'], random_state=Config.RANDOM_SEED)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=Config.data_config['val_split'], random_state=Config.RANDOM_SEED)

    train_dataset = nyu_depth_dataset(x_train, y_train, side_pixels=Config.data_config['side_pixels'], train=True)

    val_dataset  = nyu_depth_dataset(x_val,  y_val,
        side_pixels=Config.data_config['side_pixels'],
        depth_min=train_dataset.depth_min,
        depth_max=train_dataset.depth_max,
        train=False)

    test_dataset = nyu_depth_dataset(x_test, y_test,
        side_pixels=Config.data_config['side_pixels'],
        depth_min=train_dataset.depth_min,
        depth_max=train_dataset.depth_max,
        train=False)

    train_dataset_noaug = nyu_depth_dataset(x_train, y_train, side_pixels=Config.data_config['side_pixels'], train=False)



    train_loader = DataLoader(train_dataset, batch_size=Config.data_config['batch_size'], shuffle=True, num_workers=Config.data_config['num_workers_train'],
                              pin_memory=True, persistent_workers=True, prefetch_factor=2, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=Config.data_config['batch_size'], num_workers=Config.data_config['num_workers'],
                              shuffle=False, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=Config.data_config['batch_size'], num_workers=Config.data_config['num_workers'],
                              shuffle=False, pin_memory=True)

    model = get_model(Config.model_config)
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        print(f"Using {n_gpus} GPUs with DataParallel", flush=True)
        model = nn.DataParallel(model)
    model.to(device)
    optimizer = get_optimizer(model, Config.training_config)
    scheduler = get_scheduler(
    optimizer,
    Config.training_config,
    steps_per_epoch=len(train_loader)   # required for OneCycleLR
    )

    mlflow.set_tracking_uri("file://" + Config.MLFLOW_DIR)
    mlflow.set_experiment(Config.experiment_name)

    
    with mlflow.start_run(run_name=Config.run_name):

        mlflow.log_params(Config.model_config)
        mlflow.log_params(Config.training_config)
        
        mlflow.set_tag("problem", "diffusion_nyu")

        train(
            model=model,
            optimizer=optimizer,
            epochs=Config.training_config['epochs'],
            scheduler=scheduler,
            dataloader_train=train_loader,
            dataloader_val=val_loader,
            device=device,
            loss_fn_str=Config.training_config['loss'],
            weight_type=Config.training_config['weight_type'],
            side_pixels=Config.data_config['side_pixels'],
            patience=Config.training_config['patience']
        )
        test_loss = evaluate(model, test_loader, device)
        print(f"Test Loss: {test_loss:.6f}")
        mlflow.log_metric("test_loss", test_loss, step=Config.training_config['epochs'])

        m = model.module if hasattr(model, 'module') else model
        torch.save(m.state_dict(), "model_final.pth")
        mlflow.log_artifact("model_final.pth")
        
        x_base = torch.randn(1, 1, Config.data_config['side_pixels'], Config.data_config['side_pixels']).to(device)

        for ii in range(5):
            snapshots = save_flow_evolution(model, x=x_base, label=test_dataset[ii][1]/255, device=device, num_steps=1000)
            torch.save(snapshots, f"snapshots_{ii}.pt")
            mlflow.log_artifact(f"snapshots_{ii}.pt")
            create_depth_flow_animation(snapshots, filename = f"flow_evolution_log_{ii}.gif", n_steps=100, timing_mode='logarithmic')
            #create_flow_animation(snapshots, filename = f"flow_evolution_log_{ii}.gif", n_steps=100, timing_mode='logarithmic')

        for ii in range(5):
            snapshots = save_flow_evolution(model, x=x_base, label=train_dataset_noaug[ii][1]/255, device=device, num_steps=1000)
            torch.save(snapshots, f"snapshots_train_{ii}.pt")
            mlflow.log_artifact(f"snapshots_train_{ii}.pt")
            create_depth_flow_animation(snapshots, filename = f"flow_evolution_train_log_{ii}.gif", n_steps=100, timing_mode='logarithmic')

if __name__ == "__main__":
    run()