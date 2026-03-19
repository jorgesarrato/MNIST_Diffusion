import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import mlflow
import torch

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
            
        """mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        img = img * std + mean"""
        
        if img.shape[0] == 3:
            img = img.permute(1, 2, 0)
        img = img.numpy()
        
    return img

def visualize_depth_evolution_step_rgb(snapshot, gt_depth=None, downsample_factor=4, axes=None, cax=None, fig=None, log_depth=False):

    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    depth_map = snapshot['image']
    v_field = snapshot['v_field']
    rgb_condition = snapshot.get('label')

    axes[0].clear()
    if rgb_condition is not None:
        rgb_img = process_rgb_for_plot(rgb_condition)
        rgb_img = np.clip(rgb_img, 0.0, 1.0)
        axes[0].imshow(rgb_img)
        axes[0].set_title("RGB Condition")
    else:
        axes[0].text(0.5, 0.5, "No Condition", ha='center')
    axes[0].axis('off')

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
    
    depth_map_np = unscale_depth(depth_map, is_log=log_depth)

    if gt_depth is not None:
        gt_depth_np = unscale_depth(gt_depth, is_log=log_depth)
        vmin = min(depth_map_np.min(), gt_depth_np.min())
        vmax = max(depth_map_np.max(), gt_depth_np.max())
    else:
        vmin = depth_map_np.min()
        vmax = depth_map_np.max()

    axes[1].clear()
    if gt_depth is not None:
        axes[1].imshow(gt_depth_np, cmap='inferno', origin='upper', vmin=vmin, vmax=vmax)
        axes[1].set_title("Ground Truth Depth (m)")
    else:
        axes[1].text(0.5, 0.5, "No GT Depth Provided", ha='center')
    axes[1].axis('off')

    axes[2].clear()
    im_pred = axes[2].imshow(depth_map_np, cmap='inferno', origin='upper', vmin=vmin, vmax=vmax)

    if v_field is not None:
        def block_mean(ar, fact):
            h, w = ar.shape
            h_crop = (h // fact) * fact
            w_crop = (w // fact) * fact
            ar_cropped = ar[:h_crop, :w_crop]
            return ar_cropped.reshape(h_crop // fact, fact, w_crop // fact, fact).mean(axis=(1, 3))

        if isinstance(v_field, torch.Tensor):
            v_field = v_field.detach().cpu().squeeze().numpy()

        v_field_small = block_mean(v_field, downsample_factor)
        dy, dx = np.gradient(v_field_small)

        h_s, w_s = v_field_small.shape
        y = np.arange(h_s) * downsample_factor + downsample_factor / 2 - 0.5
        x = np.arange(w_s) * downsample_factor + downsample_factor / 2 - 0.5
        X, Y = np.meshgrid(x, y)

        axes[2].quiver(X, Y, dx, -dy, color='white', alpha=0.6, scale=None, width=0.005, pivot='mid')

    axes[2].set_title(f'Depth Reconstruction (t = {snapshot["t"]:.2f})')
    axes[2].axis('off')

    if fig is not None and cax is not None:
        cax.clear()
        fig.colorbar(im_pred, cax=cax, orientation='vertical')

    return axes

def create_depth_flow_animation(snapshots, filename='depth_evolution.gif', timing_mode='linear', n_steps=-1, downsample_factor=10, gt_depth=None, log_depth=False):
    total_snaps = len(snapshots)
    if (n_steps <= 0) or (n_steps > total_snaps):
        n_steps = total_snaps

    if timing_mode == 'linear':
        indices = [int(i * total_snaps / n_steps) for i in range(n_steps)]
    elif timing_mode == 'quadratic':
        indices = [int(i**2 * total_snaps / n_steps**2) for i in range(n_steps)]
    elif timing_mode == 'inv_quadratic':
        indices = [int((i / (n_steps - 1))**0.5 * (total_snaps - 1)) for i in range(n_steps)]
    elif timing_mode == 'logarithmic':
        raw = total_snaps - np.logspace(0, np.log10(total_snaps), n_steps, dtype=int)
        indices = [int(i) for i in reversed(raw)]
    else:
        raise ValueError(f"Timing mode {timing_mode} not supported.")

    indices = [min(i, total_snaps - 1) for i in indices]
    selected_snapshots = [snapshots[i] for i in indices]

    fig = plt.figure(figsize=(16, 5), dpi=100)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.03], wspace=0.05)
    
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])
    cax = fig.add_subplot(gs[3])
    axes = [ax0, ax1, ax2]
    
    def update(i):
        visualize_depth_evolution_step_rgb(
            selected_snapshots[i], 
            gt_depth=gt_depth, 
            downsample_factor=downsample_factor, 
            axes=axes, 
            cax=cax, 
            fig=fig,
            log_depth=log_depth
        )

    anim = FuncAnimation(fig, update, frames=len(selected_snapshots), interval=100)
    anim.save(filename, writer='pillow')
    plt.close()
    
    print(f"Saved animation to {filename}")
    
    if mlflow.active_run():
        mlflow.log_artifact(filename)

    return filename