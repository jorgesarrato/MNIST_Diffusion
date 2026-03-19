import os
import torch
from sklearn.model_selection import train_test_split

from utils.config import Config
from utils.readers import load_sun_rgbd_subset
from utils.model_parser import get_model
from utils.datasets import sun_depth_dataset
from evolve import save_full_flow_evolution
from utils.visualize import create_depth_flow_animation

def generate_visuals():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    torch.manual_seed(Config.RANDOM_SEED)
    print("Loading dataset paths...")
    x_paths, y_paths = load_sun_rgbd_subset(Config.SUNRGBD_DATA_DIR)

    x_train, x_test, y_train, y_test = train_test_split(
        x_paths, y_paths, test_size=Config.data_config['val_split'], random_state=Config.RANDOM_SEED
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=Config.data_config['val_split'], random_state=Config.RANDOM_SEED
    )

    train_dataset = sun_depth_dataset(x_train, y_train, cache_size=Config.data_config['cache_size'], transform=None)
    test_dataset  = sun_depth_dataset(x_test, y_test, cache_size=Config.data_config['cache_size'], transform=None)

    model_path = "model_final.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not find {model_path}. Did training finish successfully?")

    print("Loading model architecture...")
    model = get_model(Config.model_config).to(device)
    
    print(f"Loading weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
    model.load_state_dict(state_dict)
    model.eval()

    guidance_scales = Config.training_config.get('guidance_scale', 1.5)
    if not isinstance(guidance_scales, (list, tuple)):
        guidance_scales = [guidance_scales]

    log_depth_flag = Config.training_config.get('log_depth', False)
    
    os.makedirs("visualizations", exist_ok=True)
    print("Starting generation...")

    with torch.no_grad():
        for ii in range(5):
            for gs in guidance_scales:
                print(f"Generating Test Image {ii} (Scale {gs})...")
                gt_depth, rgb_label, mask = test_dataset[ii] 
                
                snapshots = save_full_flow_evolution(
                    model=model, 
                    label=rgb_label, 
                    device=device, 
                    num_steps=100, 
                    patch_size=Config.data_config['side_pixels'], 
                    stride=64,     
                    guidance_scale=gs
                )
                
                torch.save(snapshots, f"visualizations/snapshots_test_{ii}_w{gs}.pt")
                
                create_depth_flow_animation(
                    snapshots, 
                    filename=f"visualizations/flow_evolution_test_log_{ii}_w{gs}.gif", 
                    n_steps=100, 
                    timing_mode='logarithmic', 
                    gt_depth=gt_depth.unsqueeze(0), 
                    log_depth=log_depth_flag
                )

        for ii in range(5):
            for gs in guidance_scales:
                print(f"Generating Train Image {ii} (Scale {gs})...")
                gt_depth, rgb_label, mask = train_dataset[ii] 
                
                snapshots = save_full_flow_evolution(
                    model=model, 
                    label=rgb_label, 
                    device=device, 
                    num_steps=100, 
                    patch_size=Config.data_config['side_pixels'], 
                    stride=64, 
                    guidance_scale=gs
                )
                
                torch.save(snapshots, f"visualizations/snapshots_train_{ii}_w{gs}.pt")
                
                create_depth_flow_animation(
                    snapshots, 
                    filename=f"visualizations/flow_evolution_train_log_{ii}_w{gs}.gif", 
                    n_steps=100, 
                    timing_mode='logarithmic', 
                    gt_depth=gt_depth.unsqueeze(0),
                    log_depth=log_depth_flag
                )

    print("\nVisualizations saved successfully in the './visualizations/' folder.")

if __name__ == "__main__":
    generate_visuals()