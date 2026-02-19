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
from evolve import save_flow_evolution
from utils.visualize import create_depth_flow_animation

device = 'cuda' if torch.cuda.is_available() else 'cpu'

x_train, y_train = load_nyu_labeled_subset(os.path.join(Config.NYU_DATA_DIR, 'nyu_depth_v2_labeled.mat'), n_read=100)

        
dataset = nyu_depth_dataset(x_train, y_train, train = True)
dataloader = DataLoader(dataset, batch_size=4)

for x, y in dataloader:
    print(x.shape)
    print(y.shape)
    break


model = UNet_FM_Residuals([64, 128, 256], [64, 128, 256], [64, 128, 256], 256, 256, 128, in_channels=1, in_channels_cond=3)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.5, patience = 5, threshold = 0.001)


train(model, optimizer, 100, scheduler, dataloader, device)

x_base = torch.randn(1, 1, 128, 128).to(device)

labels = dataset.__getitem__(3)[1]


snapshots = save_flow_evolution(model, x=x_base, label=labels, device=device, num_steps=500)
torch.save(snapshots, f"snapshots_nyu_test.pt")
create_depth_flow_animation(snapshots, filename = f"flow_evolution_log_nyu_test.gif", n_steps=100, timing_mode='logarithmic', downsample_factor=10)
