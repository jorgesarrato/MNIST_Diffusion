from utils.readers import *
from utils.config import Config
from utils.datasets import mnist_dataset, nyu_depth_dataset
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
from models import UNet_FM
from train_FM import train
import os
from evolve import save_flow_evolution
from utils.visualize import create_depth_flow_animation

side_pixels = 128

device = 'cuda' if torch.cuda.is_available() else 'cpu'

x_base = torch.randn(1, 1, side_pixels, side_pixels).to(device)

x_train, y_train = load_nyu_labeled_subset(os.path.join(Config.NYU_DATA_DIR, 'nyu_depth_v2_labeled.mat'), n_read=1)
        
dataset = nyu_depth_dataset(x_train, y_train, train = False, side_pixels=side_pixels)
dataloader = DataLoader(dataset, batch_size=1)

model = UNet_FM([64, 128, 256], [64, 128, 256], [64, 128, 256], 256, 256, 128, in_channels=1, in_channels_cond=3, use_residuals = True)
optimizer = optim.Adam(model.parameters(), lr=0.0003)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.5, patience = 100, threshold = 0.001)

import time
t1 = time.time()
train(model, optimizer, 3000, scheduler, dataloader, device, overfit_x0=x_base)
t2 = time.time()
print(t2-t1)

labels = dataset[0][1]

snapshots = save_flow_evolution(model, x=x_base, label=labels, device=device, num_steps=500)
torch.save(snapshots, f"snapshots_nyu_test.pt")
create_depth_flow_animation(snapshots, filename = f"flow_evolution_log_nyu_test.gif", n_steps=100, timing_mode='logarithmic', downsample_factor=10)
