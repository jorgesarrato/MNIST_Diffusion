import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import mlflow
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable

def visualize_flow_step(snapshot, downsample_factor=4, axes=None):
    if axes is None:
        fig, ax = plt.subplots()
    else:
        ax = axes

    image = snapshot['image']
    v_field = snapshot['v_field']

    ax.imshow(image, cmap='magma', origin='upper')

    if v_field is not None:
        def block_mean(ar, fact):
            h, w = ar.shape
            h_crop = (h // fact) * fact
            w_crop = (w // fact) * fact
            ar_cropped = ar[:h_crop, :w_crop]
            
            return ar_cropped.reshape(h_crop // fact, fact, w_crop // fact, fact).mean(axis=(1, 3))

        v_field_small = block_mean(v_field, downsample_factor)

        dy, dx = np.gradient(v_field_small)

        h_s, w_s = v_field_small.shape
        
        y = np.arange(h_s) * downsample_factor + downsample_factor / 2 - 0.5
        x = np.arange(w_s) * downsample_factor + downsample_factor / 2 - 0.5
        X, Y = np.meshgrid(x, y)


        ax.quiver(X, Y, dx, -dy, 
                  color='white', alpha=0.8, scale=5.0, width=0.01, pivot='mid')

    ax.set_title(f't = {snapshot["t"]:.2f}')

    return ax


def create_flow_animation(snapshots, filename='flow_evolution.gif', timing_mode = 'linear', n_steps = -1):
    total_snaps = len(snapshots)
    if (n_steps <= 0) or (n_steps > total_snaps):
        n_steps = total_snaps

    if timing_mode == 'linear':
        snapshots = [snapshots[int(i*total_snaps/n_steps)] for i in range(n_steps)]
    elif timing_mode == 'quadratic':
        snapshots = [snapshots[int(i**2*total_snaps/n_steps**2)] for i in range(n_steps)]
    elif timing_mode == 'inv_quadratic':
        snapshots = [snapshots[int((i / (n_steps - 1))**0.5 * (total_snaps - 1))] for i in range(n_steps)]
    elif timing_mode == 'logarithmic':
        indices = total_snaps - np.logspace(0, np.log10(total_snaps), n_steps, dtype = int)
        snapshots = [snapshots[i] for i in reversed(indices)]
    else:
        raise ValueError(f"Timing mode {timing_mode} not supported.")

    fig, ax = plt.subplots(figsize=(6, 6))
    
    def update(i):
        ax.clear()
        visualize_flow_step(snapshots[i], axes=ax)

    anim = FuncAnimation(fig, update, frames=total_snaps, interval=100)
    anim.save(filename, writer='pillow')
    plt.close()
    print(f"Saved animation to {filename}")
    if mlflow.active_run():
        mlflow.log_artifact(filename)

    return filename

def create_multi_model_flow_animation(model_snapshots_dict, model_labels = None, filename='models_comparison.gif', timing_mode='linear', n_steps=-1, downsample_factor=4):

    model_names = list(model_snapshots_dict.keys())
    if model_labels is None:
        model_labels = model_names
    num_models = len(model_names)
    num_samples = len(model_snapshots_dict[model_names[0]])
    
    sample_zero = model_snapshots_dict[model_names[0]][0]
    total_available = len(sample_zero)
    
    if (n_steps <= 0) or (n_steps > total_available):
        n_steps = total_available

    if timing_mode == 'linear':
        indices = [int(i * (total_available - 1) / (n_steps - 1)) for i in range(n_steps)]
    elif timing_mode == 'quadratic':
        indices = [int((i**2) * (total_available - 1) / (n_steps**2)) for i in range(n_steps)]
    elif timing_mode == 'inv_quadratic':
        indices = [int((i / (n_steps - 1))**0.5 * (total_available - 1)) for i in range(n_steps)]
    elif timing_mode == 'logarithmic':
        indices = total_available - np.logspace(0, np.log10(total_available), n_steps, dtype = int)
        indices = [i for i in reversed(indices)]
    else:
        raise ValueError(f"Timing mode {timing_mode} not supported.")


    synced_data = {}
    for name in model_names:
        model_samples = []
        for sample_list in model_snapshots_dict[name]:
            model_samples.append([sample_list[idx] for idx in indices])
        synced_data[name] = model_samples

    fig, axes = plt.subplots(num_models, num_samples, 
                             figsize=(3 * num_samples, 3 * num_models), 
                             squeeze=False, layout='constrained')
    
    def update(frame_idx):
        for row_idx, model_name in enumerate(model_names):
            for col_idx in range(num_samples):
                ax = axes[row_idx, col_idx]
                ax.clear()
                
                snapshot = synced_data[model_name][col_idx][frame_idx]
                visualize_flow_step(snapshot, downsample_factor=downsample_factor, axes=ax)
                
                ax.set_xticks([])
                ax.set_yticks([])

                if col_idx == 0:
                    ax.set_ylabel(model_labels[row_idx], fontsize=12, fontweight='bold')
                
                if row_idx == 0:
                    ax.set_title(f"Sample {col_idx+1}", fontsize=10)
            
        fig.suptitle(f"Multi-Model Flow Evolution ({timing_mode} pacing) | t={snapshot['t']:.2f}", fontsize=14)

    anim = FuncAnimation(fig, update, frames=n_steps, interval=100)
    anim.save(filename, writer='pillow')
    plt.close()
    
    print(f"Grid animation saved to {filename}")
    return filename


def process_rgb_for_plot(img_tensor):
    if isinstance(img_tensor, torch.Tensor):
        img = img_tensor.detach().cpu()
        if img.ndim == 4:
            img = img.squeeze(0)
        if img.shape[0] == 3:
            img = img.permute(1, 2, 0)
        img = img.numpy()
    return img

def unscale_depth(d, d_min=0.7, d_max=10.0, is_log=True):
    if isinstance(d, torch.Tensor):
        d = d.detach().cpu().squeeze().numpy().copy()
    else:
        d = np.array(d).copy().squeeze()
        
    d = (d + 1.0) / 2.0
    
    if is_log:
        log_min, log_max = np.log(d_min), np.log(d_max)
        d_log = d * (log_max - log_min) + log_min
        return np.exp(d_log)
    else:
        return d * (d_max - d_min) + d_min

def visualize_depth_evolution_step_rgb(snapshot, gt_depth=None, mask=None, downsample_factor=10, axes=None, cax_depth=None, cax_res=None, fig=None, log_depth=False, plot_quiver=True):
    depth_map = snapshot['image']
    v_field = snapshot.get('v_field')
    rgb_condition = snapshot.get('label')

    axes[0].clear()
    if rgb_condition is not None:
        rgb_img = process_rgb_for_plot(rgb_condition)
        rgb_img = np.clip(rgb_img, 0.0, 1.0)
        axes[0].imshow(rgb_img)
        axes[0].set_title("RGB Condition")
    axes[0].axis('off')

    cmap_jet = plt.get_cmap('jet').copy()
    cmap_jet.set_bad('black')
    cmap_seismic = plt.get_cmap('seismic').copy()
    cmap_seismic.set_bad('black')

    depth_map_np = unscale_depth(depth_map, is_log=log_depth)
    
    gt_depth_np = None
    mask_np = None
    if gt_depth is not None and mask is not None:
        gt_depth_np = unscale_depth(gt_depth, is_log=log_depth)
        mask_np = mask.detach().cpu().squeeze().numpy()

        valid_pred = depth_map_np[mask_np > 0]
        valid_gt = gt_depth_np[mask_np > 0]
        
        vmin_pred = np.nanmin(valid_pred) if len(valid_pred) > 0 else 0
        vmax_pred = np.nanmax(valid_pred) if len(valid_pred) > 0 else 10
        vmin_gt = np.nanmin(valid_gt) if len(valid_gt) > 0 else 0
        vmax_gt = np.nanmax(valid_gt) if len(valid_gt) > 0 else 10
        
        vmin = min(vmin_pred, vmin_gt)
        vmax = max(vmax_pred, vmax_gt)
        
        if vmin == vmax:
            vmax += 1e-5
            vmin -= 1e-5
        
        gt_masked = np.where(mask_np > 0, gt_depth_np, np.nan)
        residuals = np.where(mask_np > 0, depth_map_np - gt_depth_np, np.nan)
    else:
        vmin = np.nanmin(depth_map_np)
        vmax = np.nanmax(depth_map_np)
        if vmin == vmax:
            vmax += 1e-5
            vmin -= 1e-5
        gt_masked = None
        residuals = np.zeros_like(depth_map_np)

    axes[1].clear()
    if gt_masked is not None:
        axes[1].imshow(gt_masked, cmap=cmap_jet, origin='upper', vmin=vmin, vmax=vmax)
        axes[1].set_title("True Depth (m)")
    axes[1].axis('off')

    axes[2].clear()
    im_pred = axes[2].imshow(depth_map_np, cmap=cmap_jet, origin='upper', vmin=vmin, vmax=vmax)
    
    if plot_quiver and v_field is not None:
        if isinstance(v_field, torch.Tensor):
            v_field = v_field.detach().cpu().squeeze().numpy()
            
        def block_mean(ar, fact):
            h, w = ar.shape
            h_crop, w_crop = (h // fact) * fact, (w // fact) * fact
            return ar[:h_crop, :w_crop].reshape(h_crop // fact, fact, w_crop // fact, fact).mean(axis=(1, 3))

        v_field_small = block_mean(v_field, downsample_factor)
        dy, dx = np.gradient(v_field_small) 
        
        h_s, w_s = v_field_small.shape
        y = np.arange(h_s) * downsample_factor + downsample_factor / 2
        x = np.arange(w_s) * downsample_factor + downsample_factor / 2
        X, Y = np.meshgrid(x, y)

        axes[2].quiver(X, Y, dx, -dy, color='white', alpha=0.5, scale=None, width=0.003, pivot='mid')

    axes[2].set_title(f'Pred Depth (t = {snapshot["t"]:.2f})')
    axes[2].axis('off')

    axes[3].clear()
    max_res = max(abs(vmin), abs(vmax))
    im_res = axes[3].imshow(residuals, cmap=cmap_seismic, origin='upper', vmin=-max_res, vmax=max_res)
    axes[3].set_title("Residuals (Pred - True)")
    axes[3].axis('off')

    if fig is not None and cax_depth is not None and cax_res is not None:
        cax_depth.clear()
        cax_res.clear()
        fig.colorbar(im_pred, cax=cax_depth, label="Depth (m)")
        fig.colorbar(im_res, cax=cax_res, label="Error (m)")

    return axes

def create_depth_flow_animation(snapshots, filename='depth_evolution.gif', timing_mode='linear', n_steps=-1, downsample_factor=10, gt_depth=None, mask=None, log_depth=False, plot_quiver=True):
    total_snaps = len(snapshots)
    if total_snaps == 0:
        print("Warning: Snapshots list is empty. Skipping animation.")
        return None
        
    if (n_steps <= 0) or (n_steps > total_snaps):
        n_steps = total_snaps

    if timing_mode == 'linear':
        indices = [int(i * total_snaps / n_steps) for i in range(n_steps)]
    elif timing_mode == 'logarithmic':
        raw = total_snaps - np.logspace(0, np.log10(max(1, total_snaps)), n_steps)
        indices = sorted([int(max(0, min(total_snaps - 1, i))) for i in raw])
    else:
        indices = [int(i * total_snaps / n_steps) for i in range(n_steps)]

    indices = [min(max(i, 0), total_snaps - 1) for i in indices]
    selected_snapshots = [snapshots[i] for i in indices]

    fig = plt.figure(figsize=(22, 5), dpi=100)
    gs = fig.add_gridspec(1, 7, width_ratios=[1, 1, 1, 1, 0.03, 0.05, 0.03], wspace=0.15)
    
    axes = [fig.add_subplot(gs[i]) for i in range(4)]
    cax_depth = fig.add_subplot(gs[4])
    cax_res = fig.add_subplot(gs[6])
    
    def update(i):
        try:
            visualize_depth_evolution_step_rgb(
                selected_snapshots[i], gt_depth=gt_depth, mask=mask, 
                downsample_factor=downsample_factor, axes=axes, 
                cax_depth=cax_depth, cax_res=cax_res, fig=fig, 
                log_depth=log_depth, plot_quiver=plot_quiver
            )
        except Exception as e:
            import traceback
            print(f"\nCRITICAL ANIMATION ERROR AT FRAME {i}:")
            traceback.print_exc()
            raise e

    anim = FuncAnimation(fig, update, frames=len(selected_snapshots), interval=100)
    anim.save(filename, writer='pillow')
    plt.close()
    
    if mlflow.active_run():
        mlflow.log_artifact(filename)
    return filename

def plot_uncertainty_stats(image, gt_depth, mask, median_depth, std_depth, filename, log_depth=False):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    rgb_img = np.clip(process_rgb_for_plot(image), 0, 1)
    axes[0].imshow(rgb_img)
    axes[0].set_title("RGB Image")
    axes[0].axis('off')
    
    gt_np = unscale_depth(gt_depth, is_log=log_depth)
    median_np = unscale_depth(median_depth, is_log=log_depth)
    std_np = unscale_depth(std_depth, is_log=log_depth) 
    
    mask_np = mask.detach().cpu().squeeze().numpy()
    
    cmap_jet = plt.get_cmap('jet').copy()
    cmap_jet.set_bad('black')
    
    gt_masked = np.where(mask_np > 0, gt_np, np.nan)
    median_masked = np.where(mask_np > 0, median_np, np.nan)
    std_masked = np.where(mask_np > 0, std_np, np.nan)
    
    vmin = min(np.nanmin(gt_masked), np.nanmin(median_masked))
    vmax = max(np.nanmax(gt_masked), np.nanmax(median_masked))
    
    axes[1].imshow(gt_masked, cmap=cmap_jet, vmin=vmin, vmax=vmax)
    axes[1].set_title("True Depth")
    axes[1].axis('off')
    
    im_med = axes[2].imshow(median_masked, cmap=cmap_jet, vmin=vmin, vmax=vmax)
    axes[2].set_title("Median Predicted Depth")
    axes[2].axis('off')
    
    divider2 = make_axes_locatable(axes[2])
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im_med, cax=cax2)
    
    im_std = axes[3].imshow(std_masked, cmap='plasma')
    axes[3].set_title("1-Sigma Uncertainty")
    axes[3].axis('off')
    
    divider3 = make_axes_locatable(axes[3])
    cax3 = divider3.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im_std, cax=cax3)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    if mlflow.active_run():
        mlflow.log_artifact(filename)

def plot_calibration_curve(all_samples_list, all_gts_list, all_masks_list, filename):
    num_imgs = len(all_samples_list)
    num_samples = all_samples_list[0].shape[0]
    
    collected_samples = []
    collected_gts = []

    for i in range(num_imgs):
        mask_i = all_masks_list[i] > 0
        
        samples_i = all_samples_list[i][:, mask_i] 
        gt_i = all_gts_list[i][mask_i]
        
        collected_samples.append(samples_i.T) 
        collected_gts.append(gt_i)

    samples_flat = np.concatenate(collected_samples, axis=0) 
    gt_flat = np.concatenate(collected_gts, axis=0)         
    total_valid_pixels = samples_flat.shape[0]
    
    percentiles = np.linspace(1, 99, 99)
    expected_probs = percentiles / 100.0
    
    quantiles = np.percentile(samples_flat, percentiles, axis=1) 
    
    observed_probs = np.mean(gt_flat[None, :] <= quantiles, axis=1)

    area_error = np.trapz(np.abs(expected_probs - observed_probs), expected_probs)

    if mlflow.active_run():
        mlflow.log_metric("calibration_area_error", float(area_error))

    data_filename = filename.replace(".png", "_data.npz")
    np.savez(data_filename, expected=expected_probs, observed=observed_probs)
    
    plt.figure(figsize=(7, 7))
    plt.plot(expected_probs, observed_probs, label="Model Calibration", color='blue', linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle='--', color='red', alpha=0.6, label="Ideal (Perfect Calibration)")
    
    textstr = '\n'.join((
        f'Images: {num_imgs}',
        f'Samples/Pixel: {num_samples}',
        f'Valid Pixels: {total_valid_pixels:,}',
        f'Calibration Error: {area_error:.4f}'
    ))
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    plt.text(0.55, 0.35, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.xlabel("Predicted Quantile (Confidence Level)")
    plt.ylabel("Observed Frequency of Ground Truth")
    plt.title("Reliability Diagram: Depth Posterior Calibration")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    
    plt.text(0.05, 0.9, "Under-confident (Above line)", color='gray', fontsize=9)
    plt.text(0.65, 0.05, "Over-confident (Below line)", color='gray', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    if mlflow.active_run():
        mlflow.log_artifact(filename)

    return expected_probs, observed_probs