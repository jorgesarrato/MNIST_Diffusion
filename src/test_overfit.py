from utils.read_MNIST import *
from utils.config import Config
from torch.utils.data import DataLoader, Dataset
import torch
import torch.optim as optim
import torch.nn as nn
from models import UNet_FM
from train_FM import train
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

x_train = load_mnist_images(os.path.join(Config.DATA_DIR, 'train-images-idx3-ubyte'))[:10]
y_train = load_mnist_labels(os.path.join(Config.DATA_DIR, 'train-labels-idx1-ubyte'))[:10]

class overfit_dataset(torch.utils.data.Dataset):
    def __init__(self, x_train, y_train):
        self.x_train = torch.tensor(x_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.float32)

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        return self.x_train[idx], 0
        
dataset = overfit_dataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size=1)

model = UNet_FM(1, [64, 128, 256], 128)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.5, patience = 3, threshold = 0.001)


train(model, optimizer, 30, scheduler, dataloader, device)