import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import mlflow

import numpy as np
import matplotlib.pyplot as plt

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
    if (n_steps <= 0) or (n_steps > len(snapshots)):
        n_steps = len(snapshots)

    if timing_mode == 'linear':
        snapshots = [snapshots[int(i*len(snapshots)/n_steps)] for i in range(n_steps)]
    elif timing_mode == 'quadratic':
        snapshots = [snapshots[int(i**2*len(snapshots)/n_steps**2)] for i in range(n_steps)]
    elif timing_mode == 'inv_quadratic':
        snapshots = [snapshots[int((i / (n_steps - 1))**0.5 * (len(snapshots) - 1))] for i in range(n_steps)]
    elif timing_mode == 'logarithmic':
        indices = 1000 - np.logspace(0, np.log10(len(snapshots)), n_steps, dtype = int)
        snapshots = [snapshots[i] for i in reversed(indices)]
    else:
        raise ValueError(f"Timing mode {timing_mode} not supported.")

    fig, ax = plt.subplots(figsize=(6, 6))
    
    def update(i):
        ax.clear()
        visualize_flow_step(snapshots[i], axes=ax)

    anim = FuncAnimation(fig, update, frames=len(snapshots), interval=100)
    anim.save(filename, writer='pillow')
    plt.close()
    print(f"Saved animation to {filename}")
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
        indices = 1000 - np.logspace(0, np.log10(total_available), n_steps, dtype = int)
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