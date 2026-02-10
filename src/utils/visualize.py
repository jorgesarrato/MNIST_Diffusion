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
    elif timing_mode == 'exponential':
        snapshots = [snapshots[int(2**i*len(snapshots)/2**n_steps)] for i in range(n_steps)]
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