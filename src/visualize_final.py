import os
import torch
import numpy as np
import mlflow
import torch.distributed as dist

from utils.config import Config
from evolve import save_full_flow_evolution, generate_fast_samples
from utils.visualize import create_depth_flow_animation, plot_uncertainty_stats, plot_calibration_curve

def run_distributed_visualizations(model, test_dataset, device, local_rank=0, world_size=1, is_main_process=True):
    print(f"[Rank {local_rank}] Starting distributed visualization...")
    os.makedirs("visualizations", exist_ok=True)
    
    m = model.module if hasattr(model, 'module') else model
    m.eval()

    guidance_scales = Config.training_config.get('guidance_scale', 1.5)
    if not isinstance(guidance_scales, (list, tuple)):
        guidance_scales = [guidance_scales]
    
    log_depth_flag = Config.data_config.get('log_depth', False)
    
    num_anim_images = 20
    with torch.no_grad():
        for ii in range(local_rank, num_anim_images, world_size):
            for gs in guidance_scales:
                print(f"[Rank {local_rank}] Generating Animation for Image {ii+1} (Scale {gs})...")
                gt_depth, rgb_label, mask = test_dataset[ii] 
                
                snapshots = save_full_flow_evolution(
                    model=m, label=rgb_label, device=device, num_steps=300, 
                    patch_size=Config.data_config['side_pixels'], stride=64, guidance_scale=gs
                )
                
                create_depth_flow_animation(
                    snapshots, filename=f"visualizations/anim_test_{ii}_w{gs}.gif", 
                    n_steps=100, timing_mode='logarithmic', 
                    gt_depth=gt_depth.unsqueeze(0), mask=mask, log_depth=log_depth_flag, plot_quiver=False
                )

    num_uncertainty_samples = 50
    num_passes = 100
    gs = guidance_scales[0] 
    
    all_samples_list, all_gts_list, all_masks_list = [], [], []

    with torch.no_grad():
        for ii in range(local_rank, num_uncertainty_samples, world_size):
            print(f"[Rank {local_rank}] Running {num_passes} passes for Test Image {ii+1}...")
            gt_depth, rgb_label, mask = test_dataset[ii]
            
            samples = generate_fast_samples(
                model=m, label=rgb_label, device=device, num_steps=50,
                patch_size=Config.data_config['side_pixels'], stride=64, 
                guidance_scale=gs, num_samples=num_passes
            )
            
            median_depth = torch.median(samples, dim=0).values
            std_depth = torch.std(samples, dim=0)
            
            plot_uncertainty_stats(
                image=rgb_label, gt_depth=gt_depth, mask=mask, 
                median_depth=median_depth, std_depth=std_depth, 
                filename=f"visualizations/uncertainty_test_{ii}.png", log_depth=log_depth_flag
            )
            
            all_samples_list.append(samples.cpu().numpy())
            all_gts_list.append(gt_depth.squeeze().cpu().numpy())
            all_masks_list.append(mask.squeeze().cpu().numpy())

    np.savez(f"visualizations/temp_calib_rank_{local_rank}.npz", 
             samples=np.array(all_samples_list, dtype=object), 
             gts=np.array(all_gts_list, dtype=object), 
             masks=np.array(all_masks_list, dtype=object))

    if world_size > 1:
        dist.barrier()

    if is_main_process:
        print("\n[Rank 0] Aggregating data and calculating Calibration...")
        combined_samples, combined_gts, combined_masks = [], [], []
        
        for r in range(world_size):
            data = np.load(f"visualizations/temp_calib_rank_{r}.npz", allow_pickle=True)
            if len(data['samples']) > 0:
                combined_samples.extend(data['samples'])
                combined_gts.extend(data['gts'])
                combined_masks.extend(data['masks'])
            os.remove(f"visualizations/temp_calib_rank_{r}.npz")

        if combined_samples:
            plot_calibration_curve(combined_samples, combined_gts, combined_masks, "visualizations/calibration_curve.png")

        if mlflow.active_run():
            mlflow.log_artifacts("visualizations/", artifact_path="visualizations")
            
        print("[Rank 0] Visualization pipeline complete!")
