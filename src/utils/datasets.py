import torch

class mnist_dataset(torch.utils.data.Dataset):
    def __init__(self, x_train, y_train):
        self.x_train = torch.tensor(x_train, dtype=torch.float32)/255
        if self.x_train.ndim == 3:
            self.x_train = self.x_train.unsqueeze(1)
        self.y_train = torch.tensor(y_train, dtype=torch.int32)

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]