import numpy as np
import struct
import h5py
from PIL import Image
import os
import tqdm
import json

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        _, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
    return images

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
    return labels

def load_nyu_labeled_subset(filename, n_read=-1):
    with h5py.File(filename, 'r') as f:
        n = f['images'].shape[0] if n_read < 0 else n_read
        img   = f['images'][:n]
        depth = f['depths'][:n]
    img   = np.transpose(img,   (0, 1, 3, 2))
    depth = np.transpose(depth, (0, 2, 1))
    return img, depth

import os

def fast_scantree(path):
    """Recursively yield DirEntry objects for given directory."""
    for entry in os.scandir(path):
        if entry.is_dir(follow_symlinks=False):
            yield from fast_scantree(entry.path)
        else:
            yield entry

def load_sun_rgbd_fast(base_dir, depth_type="depth_bfx"): 
    rgb_paths = []
    depth_paths = []
    
    for entry in fast_scantree(base_dir):
        if entry.name.endswith(".jpg") and "/image/" in entry.path:
            rgb_paths.append(entry.path)
            
        elif entry.name.endswith(".png") and f"/{depth_type}/" in entry.path:
            depth_paths.append(entry.path)
            
    return sorted(rgb_paths), sorted(depth_paths)


def load_sun_rgbd_subset(base_dir, n_read=-1):
    if not os.path.exists("dataset_manifest.json"):
        rgb, depth = load_sun_rgbd_fast(base_dir)
        with open("dataset_manifest.json", "w") as f:
            json.dump({"rgb": rgb, "depth": depth}, f)

    with open("dataset_manifest.json", "r") as f:
        data = json.load(f)
        rgb_paths, depth_paths = data["rgb"], data["depth"]
        
    if n_read > 0:
        rgb_paths = rgb_paths[:n_read]
        depth_paths = depth_paths[:n_read]
        
    return rgb_paths, depth_paths

def load_mysun_dataset(base_dir, n_read=-1):

    image_dir = os.path.join(base_dir, 'image')
    depth_dir = os.path.join(base_dir, 'depth')
    
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    
    rgb_paths = []
    depth_paths = []
    
    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        depth_file = base_name + '.png'
        
        rgb_path = os.path.join(image_dir, img_file)
        depth_path = os.path.join(depth_dir, depth_file)
        
        if os.path.exists(depth_path):
            rgb_paths.append(rgb_path)
            depth_paths.append(depth_path)
        else:
            print(f"Warning: Dropped {img_file} because {depth_file} is missing.")
            
    if n_read > 0:
        rgb_paths = rgb_paths[:n_read]
        depth_paths = depth_paths[:n_read]
        
    return rgb_paths, depth_paths

if __name__ == '__main__':
    from config import Config
    import os
    x_train = load_mnist_images(os.path.join(Config.DATA_DIR, 'train-images-idx3-ubyte'))
    y_train = load_mnist_labels(os.path.join(Config.DATA_DIR,'train-labels-idx1-ubyte'))

    print(f"Loaded MNIST training set with shape {x_train.shape}")

    img, depth = load_nyu_labeled_subset(os.path.join(Config.NYU_DATA_DIR, 'nyu_depth_v2_labeled.mat'), n_read=10)

    print(img.shape)
    print(depth.shape)
    
