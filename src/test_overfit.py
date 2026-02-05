from utils.read_MNIST import *
from utils.config import Config
from torch.utils.data import DataLoader, Datraset
import torch

x_train = load_mnist_images(os.path.join(Config.DATA_DIR, 'train-images-idx3-ubyte'))[:10]
y_train = load_mnist_labels(os.path.join(Config.DATA_DIR, 'train-labels-idx1-ubyte'))[:10]



class overfit_dataset(torch.utils.data.Dataset):
    def __init__(self, x_train, y_train):
        self.x_train = torch.tensor(x_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.float32)

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        return self.x_train[idx], 0
        