import torch

def save_flow_evolution(model, x=None, label=None, device='cpu', num_steps=50, side_pixels=128):
    m = model.module if hasattr(model, 'module') else model
    m.eval()

    if x is None:
        x = torch.randn(1, 1, side_pixels, side_pixels).to(device)
    else:
        x = x.to(device)

    label_tensor = None
    if label is not None:
        if torch.is_tensor(label):
            label_tensor = label.to(device, dtype=torch.float32)
            if label_tensor.ndim == 3:
                label_tensor = label_tensor.unsqueeze(0)
        else:
            label_tensor = torch.tensor([label], device=device, dtype=torch.float32)

    def get_v(x_curr, t_val):
        t_tensor = torch.tensor([t_val], device=device).float()
        if label_tensor is not None:
            return m(x_curr, t_tensor, label_tensor)
        return m(x_curr, t_tensor)

    dt = 1.0 / num_steps
    snapshots = []

    with torch.no_grad():
        for i in range(num_steps + 1):
            snapshot = {}
            t_val = i / num_steps

            snapshot['image'] = x.cpu().squeeze()
            snapshot['t']     = t_val
            snapshot['label'] = label

            if i < num_steps:
                v1 = get_v(x, t_val)

                x_euler = x + v1 * dt
                t_next  = (i + 1) / num_steps
                v2      = get_v(x_euler, t_next)

                x = x + (v1 + v2) / 2.0 * dt

                snapshot['v_field'] = v1.cpu().squeeze()
            else:
                snapshot['v_field'] = None

            snapshots.append(snapshot)

    return snapshots