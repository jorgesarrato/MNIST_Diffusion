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
    def __init__(self, x, y, side_pixels=128, depth_min=None, depth_max=None):
        images_raw = x.astype(np.float32)
        depths_raw = y.astype(np.float32)
        
        mask = depths_raw > 0.01
        self.depth_min = float(depths_raw[mask].min()) if depth_min is None else depth_min
        self.depth_max = float(depths_raw[mask].max()) if depth_max is None else depth_max

        print(f"Pre-processing and caching {len(images_raw)} images into RAM...")
        self.images_cache = []
        self.depths_cache = []
        resize = v2.Resize((side_pixels, side_pixels), antialias=True)
        
        for i in range(len(images_raw)):
            img = torch.from_numpy(images_raw[i]) / 255.0
            depth = torch.from_numpy(depths_raw[i]).unsqueeze(0)

            depth = torch.clamp(depth, self.depth_min, self.depth_max)
            depth = (depth - self.depth_min) / (self.depth_max - self.depth_min)
            depth = depth * 2.0 - 1.0

            stacked = torch.cat([img, depth], dim=0)
            stacked = resize(stacked)
            
            self.images_cache.append(stacked[:3])
            self.depths_cache.append(stacked[3:4])

    def __len__(self):
        return len(self.images_cache)
    
    def __getitem__(self, idx):
        # Now getitem is instantaneous!
        return self.depths_cache[idx], self.images_cache[idx]