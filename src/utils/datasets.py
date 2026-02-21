import torch
from torchvision.transforms import v2
import numpy as np
import torch.nn.functional as F

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
    def __init__(self, x, y, train=True, side_pixels=128):
        self.images = x.astype(np.float32)
        self.depths = y.astype(np.float32)
        self.side_pixels = side_pixels
        self.train = train

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        img = torch.from_numpy(self.images[idx])
        depth = torch.from_numpy(self.depths[idx]).unsqueeze(0)

        depth_min, depth_max = depth.min(), depth.max()
        if depth_max - depth_min > 0:
            depth = (depth - depth_min) / (depth_max - depth_min) # to [0, 1]
        depth = (depth * 2.0) - 1.0 # to [-1, 1]

        img = img / 255.0

        if self.train:
            stacked = torch.cat([img, depth], dim=0)
            
            stacked = v2.RandomHorizontalFlip(p=0.5)(stacked)
            stacked = v2.RandomResizedCrop(
                size=(self.side_pixels, self.side_pixels), 
                scale=(0.8, 1.0),
                antialias=True
            )(stacked)
            
            img, depth = torch.split(stacked, [3, 1], dim=0)
            
            img = v2.ColorJitter(brightness=0.2, contrast=0.2)(img)
        else:
            img = v2.Resize((self.side_pixels, self.side_pixels), antialias=True)(img)
            depth = v2.Resize((self.side_pixels, self.side_pixels), antialias=True)(depth)

        return depth, img
        
