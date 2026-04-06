import os
import math
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import mlflow
from mlflow.tracking import MlflowClient
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.transforms import v2
from PIL import Image

from utils.config import Config
from utils.readers import load_mysun_dataset
from utils.datasets import sun_depth_dataset
from utils.model_parser import get_model
from evolve import save_full_flow_evolution, generate_fast_samples
from sklearn.model_selection import train_test_split

def to_metric_depth(x, is_log, d_min=0.7, d_max=10.0):
    if is_log:
        l_min, l_max = math.log(d_min), math.log(d_max)
        return torch.exp((x + 1.0) / 2.0 * (l_max - l_min) + l_min)
    else:
        return (x + 1.0) / 2.0 * (d_max - d_min) + d_min

def compute_depth_metrics(pred, gt, mask):
    """Computes standard depth estimation metrics."""
    valid = mask > 0
    if not np.any(valid):
        return None
    
    p = pred[valid]
    g = gt[valid]
    
    thresh = np.maximum((g / p), (p / g))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25**2).mean()
    a3 = (thresh < 1.25**3).mean()
    
    abs_rel = np.mean(np.abs(g - p) / g)
    sq_rel = np.mean(((g - p)**2) / g)
    rmse = np.sqrt(np.mean((g - p)**2))
    rmse_log = np.sqrt(np.mean((np.log(g) - np.log(p))**2))
    
    return {"abs_rel": abs_rel, "sq_rel": sq_rel, "rmse": rmse, "rmse_log": rmse_log, "a1": a1, "a2": a2, "a3": a3}

def get_dinov2_runs(client, experiment_id):
    query = "tags.mlflow.runName LIKE 'DINOv2%'"
    runs = client.search_runs(experiment_ids=[experiment_id], filter_string=query)
    return sorted(runs, key=lambda r: len(r.data.tags.get("mlflow.runName", r.info.run_id)))

def create_superimposed_calibration(run_names, run_data, output_path):
    plt.figure(figsize=(8, 8))
    for name, data in zip(run_names, run_data):
        if data is not None and 'expected' in data:
            area_error = np.trapz(np.abs(data['observed'] - data['expected']), data['expected'])
            plt.plot(data['expected'], data['observed'], label=f"{name} (AE: {area_error:.3f})", linewidth=2)
            
    plt.plot([0, 1], [0, 1], linestyle='--', color='red', alpha=0.6, label="Ideal")
    plt.xlabel("Predicted Quantile")
    plt.ylabel("Observed Frequency")
    plt.title("Calibration Comparison: DINOv2 Models")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def create_model_comparison_grid(rgb, gt_metric, mask, models_data, sample_idx, out_path):
    num_models = len(models_data)
    fig, axes = plt.subplots(num_models + 1, 3, figsize=(12, 4 * (num_models + 1)))
    
    cmap_jet = copy.copy(cm.jet)
    cmap_jet.set_bad(color='black')
    
    cmap_inf = copy.copy(cm.inferno)
    cmap_inf.set_bad(color='black')
    
    valid_mask = mask.squeeze().cpu().numpy() > 0
    gt_np = gt_metric.squeeze().cpu().numpy().astype(float)
    
    vmin = gt_np[valid_mask].min() if valid_mask.any() else 0.7
    vmax = gt_np[valid_mask].max() if valid_mask.any() else 10.0
    
    gt_np[~valid_mask] = np.nan
    
    # Dynamically calculate the smallest 99th percentile std across all models
    std_p99_list = []
    for run_name, data in models_data.items():
        std_np = data['std'].squeeze().astype(float)
        if valid_mask.any():
            std_p99_list.append(np.percentile(std_np[valid_mask], 99))
            
    std_vmax = min(std_p99_list) if std_p99_list else 1.5

    axes[0, 0].imshow(rgb.permute(1, 2, 0).cpu().numpy())
    axes[0, 0].set_title("RGB Image")
    
    axes[0, 1].imshow(gt_np, cmap=cmap_jet, vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(f"Ground Truth Depth (m)\nScale: {vmin:.1f}m - {vmax:.1f}m")
    
    axes[0, 2].axis('off')
    axes[0, 2].text(0.5, 0.5, f"Shared Uncertainty Scale:\n0.00m - {std_vmax:.2f}m", 
                    fontsize=12, ha='center', va='center', wrap=True)
    
    for i, (run_name, data) in enumerate(models_data.items()):
        row = i + 1
        axes[row, 0].text(0.5, 0.5, run_name, fontsize=12, ha='center', va='center', wrap=True)
        axes[row, 0].axis('off')
        
        med_np = data['median'].squeeze().astype(float)
        med_np[~valid_mask] = np.nan
        
        im_med = axes[row, 1].imshow(med_np, cmap=cmap_jet, vmin=vmin, vmax=vmax)
        axes[row, 1].set_title("Median Depth (m)")
        plt.colorbar(im_med, ax=axes[row, 1], fraction=0.046, pad=0.04)
        
        std_np = data['std'].squeeze().astype(float)
        std_np[~valid_mask] = np.nan
        
        # Apply the new dynamically calculated saturation limit
        im_std = axes[row, 2].imshow(std_np, cmap=cmap_inf, vmin=0.0, vmax=std_vmax)
        axes[row, 2].set_title("Uncertainty Std (m)")
        plt.colorbar(im_std, ax=axes[row, 2], fraction=0.046, pad=0.04)
        
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def combine_gifs_vertically(run_names, gif_paths, output_path):
    import imageio
    readers = [imageio.get_reader(p) for p in gif_paths]
    frames_list = [[frame for frame in r] for r in readers]
    
    num_frames = min(len(f) for f in frames_list)
    combined_frames = []
    
    for i in range(num_frames):
        combined_img = np.concatenate([frames[i] for frames in frames_list], axis=0)
        combined_frames.append(combined_img)
        
    imageio.mimsave(output_path, combined_frames, fps=10, loop=0)

def main():
    is_distributed = "LOCAL_RANK" in os.environ
    if is_distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)
    else:
        local_rank = 0
        world_size = 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    is_main_process = local_rank == 0
    os.makedirs("comparisons", exist_ok=True)

    x_paths, y_paths = load_mysun_dataset(Config.MYSUNRGBD_DATA_DIR)
    x_train, x_test, y_train, y_test = train_test_split(x_paths, y_paths, test_size=Config.data_config['val_split'], random_state=Config.RANDOM_SEED)
    
    test_transform_full = v2.Identity()
    test_dataset = sun_depth_dataset(x_test, y_test, cache_size=Config.data_config['cache_size'], transform=test_transform_full)

    client = MlflowClient(tracking_uri=f"file://{Config.MLFLOW_DIR}")
    experiment = client.get_experiment_by_name(Config.experiment_name)
    
    if experiment is None:
        if is_main_process: print("Experiment not found!")
        return

    runs = get_dinov2_runs(client, experiment.experiment_id)
    if is_main_process: print(f"Found {len(runs)} DINOv2 runs.")

    calib_data_all = []
    final_metrics_table = {}
    run_names = []
    target_indices = [0, 1, 2] 

    for run in runs:
        run_name = run.data.tags.get("mlflow.runName", run.info.run_id)
        run_names.append(run_name)
        run_id = run.info.run_id
        
        if is_main_process: print(f"\n--- Processing {run_name} ---")

        artifact_path = client.download_artifacts(run_id, "")
        model_path = os.path.join(artifact_path, "model_final.pth")
        local_calib_path = f"comparisons/{run_name}_calib_data.npz"
        
        if not os.path.exists(model_path):
            if is_main_process: print(f"Skipping {run_name}: No model_final.pth found.")
            calib_data_all.append(None)
            continue

        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        model_cfg = ckpt['config']['model_config']
        
        run_data_cfg = ckpt['config'].get('data_config', {})
        run_train_cfg = ckpt['config'].get('training_config', {})
        model_is_log = run_data_cfg.get('log_depth', run_train_cfg.get('log_depth', False))
        
        model = get_model(model_cfg).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        if is_distributed:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        m = model.module if hasattr(model, 'module') else model
        m.eval()

        if os.path.exists(local_calib_path):
            if is_main_process:
                print(f"Loaded existing calibration & metrics data from local folder for {run_name}.")
                data = np.load(local_calib_path)
                
                if 'abs_rel' not in data:
                    print(f"WARNING: Old {local_calib_path} found without metrics! Delete the comparisons folder to regenerate.")
                    calib_data_all.append({'expected': data['expected'], 'observed': data['observed']})
                else:
                    metrics_dict = {k: data[k].item() for k in ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]}
                    final_metrics_table[run_name] = metrics_dict
                    calib_data_all.append({'expected': data['expected'], 'observed': data['observed']})
                
        else:
            if is_main_process: print(f"Metrics missing for {run_name}. Running 50-image distributed evaluation...")
            all_samples_list, all_gts_list, all_masks_list = [], [], []
            num_passes = 100
            current_is_log = Config.data_config.get('log_depth', False)
            
            with torch.no_grad():
                for ii in range(local_rank, 50, world_size):
                    gt_depth, rgb_label, mask = test_dataset[ii]
                    gt_metric = to_metric_depth(gt_depth, current_is_log)
                    
                    samples = generate_fast_samples(
                        model=m, label=rgb_label, device=device, num_steps=50,
                        patch_size=model_cfg['side_pixels'], stride=64, guidance_scale=1.5, num_samples=num_passes
                    )
                    samples_metric = to_metric_depth(samples, model_is_log)
                    
                    all_samples_list.append(samples_metric.cpu().numpy())
                    all_gts_list.append(gt_metric.squeeze().cpu().numpy())
                    all_masks_list.append(mask.squeeze().cpu().numpy())

            samples_arr = np.empty(len(all_samples_list), dtype=object)
            gts_arr = np.empty(len(all_gts_list), dtype=object)
            masks_arr = np.empty(len(all_masks_list), dtype=object)
            for i in range(len(all_samples_list)):
                samples_arr[i] = all_samples_list[i]
                gts_arr[i] = all_gts_list[i]
                masks_arr[i] = all_masks_list[i]

            np.savez(f"comparisons/temp_{run_name}_rank_{local_rank}.npz", samples=samples_arr, gts=gts_arr, masks=masks_arr)
            
            if is_distributed: dist.barrier()
            
            if is_main_process:
                combined_samples, combined_gts, combined_masks = [], [], []
                for r in range(world_size):
                    temp_file = f"comparisons/temp_{run_name}_rank_{r}.npz"
                    if os.path.exists(temp_file):
                        data = np.load(temp_file, allow_pickle=True)
                        if len(data['samples']) > 0:
                            combined_samples.extend(data['samples'])
                            combined_gts.extend(data['gts'])
                            combined_masks.extend(data['masks'])
                        os.remove(temp_file)

                if combined_samples:
                    from utils.visualize import plot_calibration_curve
                    calib_out_png = f"comparisons/{run_name}_calib.png"
                    expected, observed = plot_calibration_curve(combined_samples, combined_gts, combined_masks, calib_out_png)
                    
                    img_metrics = []
                    for i in range(len(combined_samples)):
                        pred_median = np.median(combined_samples[i], axis=0)
                        res = compute_depth_metrics(pred_median, combined_gts[i], combined_masks[i])
                        if res: img_metrics.append(res)
                        
                    avg_metrics = {k: np.mean([m[k] for m in img_metrics]) for k in img_metrics[0].keys()}
                    
                    for k, v in avg_metrics.items():
                        client.log_metric(run_id, f"eval_50_{k}", float(v))
                        
                    final_metrics_table[run_name] = avg_metrics
                    
                    np.savez(local_calib_path, expected=expected, observed=observed, **avg_metrics)
                    calib_data_all.append({'expected': expected, 'observed': observed})
                
            if is_distributed: dist.barrier()

        with torch.no_grad():
            for ii in target_indices:
                if ii % world_size == local_rank:
                    stat_file = f"comparisons/{run_name}_sample_{ii}_stats.npz"
                    gif_path = f"comparisons/{run_name}_sample_{ii}.gif"
                    
                    gt_depth, rgb_label, mask = test_dataset[ii]
                    
                    if not os.path.exists(stat_file):
                        print(f"Generating stats for {run_name} sample {ii}...")
                        samples = generate_fast_samples(
                            model=m, label=rgb_label, device=device, num_steps=50,
                            patch_size=model_cfg['side_pixels'], stride=64, guidance_scale=1.5, num_samples=50
                        )
                        samples_metric = to_metric_depth(samples, model_is_log)
                        median_depth = torch.median(samples_metric, dim=0).values.cpu().numpy()
                        std_depth = torch.std(samples_metric, dim=0).cpu().numpy()
                        np.savez(stat_file, median=median_depth, std=std_depth)
                    else:
                        print(f"Skipping stats generation for {run_name} sample {ii} (Already exists)")
                        
                    if not os.path.exists(gif_path):
                        print(f"Generating GIF for {run_name} sample {ii}...")
                        snapshots = save_full_flow_evolution(
                            model=m, label=rgb_label, device=device, num_steps=100, 
                            patch_size=model_cfg['side_pixels'], stride=64, guidance_scale=1.5
                        )
                        from utils.visualize import create_depth_flow_animation
                        create_depth_flow_animation(
                            snapshots, filename=gif_path, n_steps=100, 
                            timing_mode='logarithmic', 
                            gt_depth=gt_depth.unsqueeze(0), mask=mask, 
                            log_depth=model_is_log, plot_quiver=False
                        )
                    else:
                        print(f"Skipping GIF generation for {run_name} sample {ii} (Already exists)")

        if is_distributed: dist.barrier()

    if is_main_process:
        print("\n--- Generating Combined Visualizations ---")
        
        create_superimposed_calibration(run_names, calib_data_all, "comparisons/all_calibration_curves.png")
        
        current_is_log = Config.data_config.get('log_depth', False)
        
        for ii in target_indices:
            gt_depth, rgb_label, mask = test_dataset[ii]
            gt_metric = to_metric_depth(gt_depth, current_is_log)
            
            models_data = {}
            gif_paths = []
            
            for run_name in run_names:
                stat_file = f"comparisons/{run_name}_sample_{ii}_stats.npz"
                if os.path.exists(stat_file):
                    data = np.load(stat_file)
                    models_data[run_name] = {'median': data['median'], 'std': data['std']}
                
                gif_path = f"comparisons/{run_name}_sample_{ii}.gif"
                if os.path.exists(gif_path):
                    gif_paths.append(gif_path)
            
            create_model_comparison_grid(rgb_label, gt_metric, mask, models_data, ii, f"comparisons/grid_sample_{ii}.png")
            
            if len(gif_paths) == len(run_names) and len(gif_paths) > 0:
                combine_gifs_vertically(run_names, gif_paths, f"comparisons/combined_evolution_{ii}.gif")
        
        if final_metrics_table:
            print("\n" + "="*85)
            print(f"{'Model (DINOv2 Variants)':<35} | {'Abs Rel':<7} | {'RMSE':<7} | {'RMSE log':<8} | {'δ<1.25':<7} | {'δ<1.25²':<7} | {'δ<1.25³':<7}")
            print("-" * 85)
            for r_name in run_names:
                if r_name in final_metrics_table:
                    m = final_metrics_table[r_name]
                    print(f"{r_name:<35} | {m['abs_rel']:<7.3f} | {m['rmse']:<7.3f} | {m['rmse_log']:<8.3f} | {m['a1']:<7.3f} | {m['a2']:<7.3f} | {m['a3']:<7.3f}")
            print("="*85 + "\n")
            
        print("Done! Check the 'comparisons' folder.")

if __name__ == "__main__":
    main()