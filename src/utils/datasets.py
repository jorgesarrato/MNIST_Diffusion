import torch
from torchvision.transforms import v2
import numpy as np
import random

class mnist_dataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)/255
        if self.x.ndim == 3:
            self.x = self.x.unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.int32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class nyu_depth_dataset(torch.utils.data.Dataset):
    def __init__(self, x, y, side_pixels=128, depth_min=None, depth_max=None, train=True):
        self.images     = x.astype(np.float32)
        self.depths     = y.astype(np.float32)
        self.side_pixels = side_pixels
        self.train = train

        mask = self.depths > 0.01
        self.depth_min = float(self.depths[mask].min()) if depth_min is None else depth_min
        self.depth_max = float(self.depths[mask].max()) if depth_max is None else depth_max

    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self, idx):
        img   = torch.from_numpy(self.images[idx])
        depth = torch.from_numpy(self.depths[idx]).unsqueeze(0)

        depth = torch.clamp(depth, self.depth_min, self.depth_max)
        depth = (depth - self.depth_min) / (self.depth_max - self.depth_min)
        depth = depth * 2.0 - 1.0

        stacked = torch.cat([img, depth], dim=0)
        if self.train:
            if random.random() < 0.5:
                stacked = v2.RandomCrop(self.side_pixels)(stacked)
            else:
                stacked = v2.Resize((self.side_pixels, self.side_pixels),
                                    antialias=True)(stacked)
        else:
            stacked = v2.Resize((self.side_pixels, self.side_pixels),
                        antialias=True)(stacked)
        
        img, depth = stacked[:3], stacked[3:4]

        return depth, img