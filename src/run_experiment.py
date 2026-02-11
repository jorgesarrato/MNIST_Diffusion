import os

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
import mlflow

from utils.config import Config
from utils.read_MNIST import load_mnist_images, load_mnist_labels
from utils.model_parser import get_model
from utils.opt_parser import get_optimizer
from utils.datasets import mnist_dataset
from utils.visualize import create_flow_animation
from train_FM import train, evaluate
from evolve import save_flow_evolution

def run():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #Set seed
    torch.manual_seed(Config.RANDOM_SEED)
    if device == 'cuda':
        torch.cuda.manual_seed_all(Config.RANDOM_SEED)


    x_train = load_mnist_images(os.path.join(Config.DATA_DIR, 'train-images-idx3-ubyte'))
    y_train = load_mnist_labels(os.path.join(Config.DATA_DIR,'train-labels-idx1-ubyte'))
    x_test = load_mnist_images(os.path.join(Config.DATA_DIR, 't10k-images-idx3-ubyte'))
    y_test = load_mnist_labels(os.path.join(Config.DATA_DIR, 't10k-labels-idx1-ubyte'))

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=Config.data_config['val_split'], random_state=Config.RANDOM_SEED)

    train_dataset = mnist_dataset(x_train, y_train)
    val_dataset = mnist_dataset(x_val, y_val)
    test_dataset = mnist_dataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=Config.data_config['batch_size'], shuffle=True, num_workers=Config.data_config['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=Config.data_config['batch_size'], num_workers=Config.data_config['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=Config.data_config['batch_size'], num_workers=Config.data_config['num_workers'])

    model = get_model(Config.model_config)
    optimizer = get_optimizer(model, Config.training_config)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        factor=Config.training_config['scheduler_factor'], 
        patience=Config.training_config['patience']
    )

    mlflow.set_experiment(Config.experiment_name)
    
    with mlflow.start_run():

        mlflow.log_params(Config.model_config)
        mlflow.log_params(Config.training_config)
        
        mlflow.set_tag("problem", "diffusion_mnist")

        train(
            model=model,
            optimizer=optimizer,
            epochs=Config.training_config['epochs'],
            scheduler=scheduler,
            dataloader_train=train_loader,
            dataloader_val=val_loader,
            device=device
        )

        test_loss = evaluate(model, test_loader, device)
        print(f"Test Loss: {test_loss:.6f}")
        mlflow.log_metric("test_loss", test_loss, step=Config.training_config['epochs'])

        torch.save(model.state_dict(), "model_final.pth")
        mlflow.log_artifact("model_final.pth")

        for ii in range(5):
            x_base = torch.randn(1, 1, 28, 28).to(device)
            snapshots = save_flow_evolution(model, x=x_base, device=device, num_steps=1000)
            torch.save(snapshots, f"snapshots_{ii}.pt")
            mlflow.log_artifact(f"snapshots_{ii}.pt")
            create_flow_animation(snapshots, filename = f"flow_evolution_linear_{ii}.gif", n_steps=100, timing_mode='linear')
            create_flow_animation(snapshots, filename = f"flow_evolution_inv_quad_{ii}.gif", n_steps=100, timing_mode='inv_quadratic')
            create_flow_animation(snapshots, filename = f"flow_evolution_log_{ii}.gif", n_steps=1, timing_mode='logarithmic')

if __name__ == "__main__":
    run()