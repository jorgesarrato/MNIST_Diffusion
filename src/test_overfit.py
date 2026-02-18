from utils.readers import *
from utils.config import Config
from utils.datasets import mnist_dataset, nyu_depth_dataset
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
from models import UNet_FM, UNet_FM_Residuals
from train_FM import train
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

x_train, y_train = load_nyu_labeled_subset(os.path.join(Config.NYU_DATA_DIR, 'nyu_depth_v2_labeled.mat'), n_read=100)

        
dataset = nyu_depth_dataset(x_train, y_train, train = True)
dataloader = DataLoader(dataset, batch_size=4)

model = UNet_FM_Residuals([32, 64, 128], [32, 64, 128], [128, 64, 32], 128, 128, 128)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, threshold = 0.001)


train(model, optimizer, 100, scheduler, dataloader, device)