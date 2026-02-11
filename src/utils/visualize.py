import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import mlflow

def visualize_flow_step(snapshot, downsample_factor = 4, axes = None):
    if axes is None:
        fig, ax = plt.subplots()
    else:
        ax = axes

    image = snapshot['image']
    v_field = snapshot['v_field']

    ax.imshow(image, cmap='magma', origin='upper')

    if v_field is not None:

        dy, dx = np.gradient(v_field)

        h, w = v_field.shape
        y, x = np.arange(h), np.arange(w)
        X, Y = np.meshgrid(x, y)

        skip = slice(None, None, downsample_factor) 
        

        ax.quiver(X[skip, skip], Y[skip, skip], dx[skip, skip], -dy[skip, skip], 
                color='white', alpha=0.8, scale=20.0, width=0.01)

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

def create_multi_model_flow_animation(model_snapshots_dict, filename='models_comparison.gif', timing_mode='linear', n_steps=-1, downsample_factor=4):

    model_names = list(model_snapshots_dict.keys())
    num_models = len(model_names)
    
    total_available = len(model_snapshots_dict[model_names[0]])
    
    if (n_steps <= 0) or (n_steps > total_available):
        n_steps = total_available

    if timing_mode == 'linear':
        indices = [int(i * (total_available - 1) / (n_steps - 1)) for i in range(n_steps)]
    elif timing_mode == 'quadratic':
        indices = [int((i**2) * (total_available - 1) / (n_steps**2)) for i in range(n_steps)]
    elif timing_mode == 'inv_quadratic':
        indices = [int((i / (n_steps - 1))**0.5 * (total_available - 1)) for i in range(n_steps)]
    elif timing_mode == 'logarithmic':
        indices = np.logspace(0, np.log10(total_available), n_steps, dtype=int) - 1
        indices = np.clip(indices, 0, total_available - 1)
    else:
        raise ValueError(f"Timing mode {timing_mode} not supported.")

    synced_snapshots = {
        name: [model_snapshots_dict[name][idx] for idx in indices] 
        for name in model_names
    }

    fig, axes = plt.subplots(num_models, 1, figsize=(6, 5 * num_models), 
                             squeeze=False, layout='constrained')
    
    def update(frame_idx):
        for row_idx, model_name in enumerate(model_names):
            ax = axes[row_idx, 0]
            ax.clear()
            
            snapshot = synced_snapshots[model_name][frame_idx]
            
            visualize_flow_step(snapshot, downsample_factor=downsample_factor, axes=ax)
            
            ax.set_ylabel(model_name, fontsize=12, fontweight='bold')
            if row_idx == 0:
                fig.suptitle(f"Flow Evolution Comparison ({timing_mode} pacing)", fontsize=14)

    anim = FuncAnimation(fig, update, frames=n_steps, interval=100)
    anim.save(filename, writer='pillow')
    plt.close()
    
    print(f"Comparison animation ({timing_mode}) saved to {filename}")

    return filename