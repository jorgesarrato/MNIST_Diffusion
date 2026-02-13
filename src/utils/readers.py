import numpy as np
import struct
import h5py

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        _, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
    return images

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
    return labels

def load_nyu_labeled_subset(filename, n_read = -1):
    with h5py.File(filename, 'r') as f:        

        img = f['images']
        depth = f['depths']

        if n_read < 0:
            n_read = len(img)
        
        img = np.transpose(img, (0, 3, 2, 1))
        
        depth = np.transpose(depth, (0, 2, 1))
        
    return img[:n_read], depth[:n_read]

if __name__ == '__main__':
    from config import Config
    import os
    x_train = load_mnist_images(os.path.join(Config.DATA_DIR, 'train-images-idx3-ubyte'))
    y_train = load_mnist_labels(os.path.join(Config.DATA_DIR,'train-labels-idx1-ubyte'))

    print(f"Loaded MNIST training set with shape {x_train.shape}")

    img, depth = load_nyu_labeled_subset(os.path.join(Config.NYU_DATA_DIR, 'nyu_depth_v2_labeled.mat'), n_read=10)
    
