import torch
from torchvision.transforms import v2
import numpy as np

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
        return self.depths_cache[idx], self.images_cache[idx]
    

class sun_depth_dataset(torch.utils.data.Dataset):
    def __init__(self, images_list, depths_list, cache_size=384, transform=None):
        self.images_cache = []
        self.depths_cache = []
        self.transform = transform
        
        standardize = v2.Compose([
            v2.Resize(cache_size, antialias=True),
            v2.CenterCrop(cache_size) 
        ])
        
        print(f"Standardizing images to {cache_size}x{cache_size} squares...")
        for img_np, depth_np in zip(images_list, depths_list):

            valid_rows = np.any(depth_np > 0, axis=1)
            valid_cols = np.any(depth_np > 0, axis=0)
            
            if not valid_rows.any() or not valid_cols.any():
                continue
                
            rmin, rmax = np.where(valid_rows)[0][[0, -1]]
            cmin, cmax = np.where(valid_cols)[0][[0, -1]]
            
            img_cropped = img_np[rmin:rmax+1, cmin:cmax+1]
            depth_cropped = depth_np[rmin:rmax+1, cmin:cmax+1]
            
            img_t = torch.from_numpy(img_cropped).permute(2, 0, 1) / 255.0
            depth_t = torch.from_numpy(depth_cropped).unsqueeze(0)
            
            depth_t = torch.clamp(depth_t, 0.7, 10.0)
            depth_t = (depth_t - 0.7) / (10.0 - 0.7)
            depth_t = depth_t * 2.0 - 1.0
            
            stacked = torch.cat([img_t, depth_t], dim=0)
            stacked = standardize(stacked)
            
            self.images_cache.append(stacked[:3])
            self.depths_cache.append(stacked[3:4])

    def __len__(self):
        return len(self.images_cache)
    
    def __getitem__(self, idx):
        depth = self.depths_cache[idx]
        img = self.images_cache[idx]
        
        if self.transform is not None:
            stacked = torch.cat([img, depth], dim=0)
            stacked = self.transform(stacked)
            img, depth = stacked[:3], stacked[3:4]
            
        return depth, img