import torch
from torchvision.transforms import v2
import numpy as np
from PIL import Image
import math

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
    import torch

class sun_depth_dataset(torch.utils.data.Dataset):
    def __init__(self, images_paths, depths_paths, cache_size=384, transform=None, log_depth=True):
        self.images_paths = images_paths
        self.depths_paths = depths_paths
        self.transform = transform
        self.log_depth = log_depth
        self.cache_size = cache_size
        
        if self.cache_size > 0:
            self.normalize_scale = v2.Resize(cache_size, interpolation=v2.InterpolationMode.BILINEAR, antialias=True)
        else:
            self.normalize_scale = None
                
        self.DEPTH_MIN, self.DEPTH_MAX = 0.7, 10.0
        self.LOG_MIN = math.log(self.DEPTH_MIN)
        self.LOG_MAX = math.log(self.DEPTH_MAX)

    def __len__(self):
        return len(self.images_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.images_paths[idx]).convert('RGB')
        depth = Image.open(self.depths_paths[idx])
        
        img_np = np.array(img)
        depth_np = np.array(depth).astype(np.uint16)
        
        depth_corrected = (depth_np >> 3) | (depth_np << 13)
        depth_corrected = depth_corrected.astype(np.float32) / 1000.0
        depth_corrected[depth_corrected > 10.0] = 0.0
        
        valid_rows = np.any(depth_corrected > 0, axis=1)
        valid_cols = np.any(depth_corrected > 0, axis=0)
        
        if valid_rows.any() and valid_cols.any():
            rmin, rmax = np.where(valid_rows)[0][[0, -1]]
            cmin, cmax = np.where(valid_cols)[0][[0, -1]]
            
            img_np = img_np[rmin:rmax+1, cmin:cmax+1]
            depth_corrected = depth_corrected[rmin:rmax+1, cmin:cmax+1]
            
        img_t = torch.from_numpy(img_np).permute(2, 0, 1) / 255.0
        depth_t = torch.from_numpy(depth_corrected).unsqueeze(0)
        
        valid_mask_t = (depth_t > 0.0).float()
        
        depth_t = torch.clamp(depth_t, self.DEPTH_MIN, self.DEPTH_MAX)
        
        if self.log_depth:
            depth_t = (torch.log(depth_t) - self.LOG_MIN) / (self.LOG_MAX - self.LOG_MIN)
        else:
            depth_t = (depth_t - self.DEPTH_MIN) / (self.DEPTH_MAX - self.DEPTH_MIN)
            
        depth_t = depth_t * 2.0 - 1.0
        
        stacked = torch.cat([img_t, depth_t, valid_mask_t], dim=0)

        if self.normalize_scale is not None:
            stacked = self.normalize_scale(stacked)
        
        if self.transform is not None:
            stacked = self.transform(stacked)

        img_out = stacked[:3]
        depth_out = stacked[3:4]
        
        valid_mask_out = (stacked[4:5] > 0.5).float()
        
        return depth_out, img_out, valid_mask_out