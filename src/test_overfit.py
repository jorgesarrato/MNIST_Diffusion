from utils.read_MNIST import *
from utils.config import Config
from utils.datasets import mnist_dataset
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
from models import UNet_FM
from train_FM import train
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

x_train = load_mnist_images(os.path.join(Config.DATA_DIR, 'train-images-idx3-ubyte'))[:100]
y_train = load_mnist_labels(os.path.join(Config.DATA_DIR, 'train-labels-idx1-ubyte'))[:100]
        
dataset = mnist_dataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size=4)

model = UNet_FM([32, 64, 128], 64)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, threshold = 0.001)


train(model, optimizer, 300, scheduler, dataloader, device)