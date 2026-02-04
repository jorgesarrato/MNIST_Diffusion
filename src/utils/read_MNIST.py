import numpy as np
import struct

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        _, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
    return images

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        _, num = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    return labels

if __name__ == '__main__':
    from config import Config
    import os
    x_train = load_mnist_images(os.path.join(Config.DATA_DIR, 'train-images-idx3-ubyte'))
    y_train = load_mnist_labels(os.path.join(Config.DATA_DIR,'train-labels-idx1-ubyte'))

    print(f"Loaded MNIST training set with shape {x_train.shape}")
