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
    def __init__(self, x, y, train = True, side_pixels = 128):
            self.images = x
            self.depths = y

            if train:
                self.transform = v2.Compose([
                    v2.RandomHorizontalFlip(p=0.5),
                    v2.RandomResizedCrop(size=(side_pixels, side_pixels), scale=(0.5, 1.0), antialias=True),
                    v2.ColorJitter(brightness=0.2, contrast=0.2)                       
                ])
            else:
                self.transform = v2.Resize((side_pixels, side_pixels), antialias=True)


    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        img = torch.from_numpy(self.images[idx])
        depth = torch.from_numpy(np.expand_dims(self.depths[idx],0))
        
        img = self.transform(img)
        depth = self.transform(depth)
        
        return depth, img
        
