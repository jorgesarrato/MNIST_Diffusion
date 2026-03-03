import torch

def save_flow_evolution(model, x=None, label=None, device='cpu', num_steps=50, side_pixels=128):
    m = model.module if hasattr(model, 'module') else model
    m.eval()

    if x is None:
        x = torch.randn(1, 1, side_pixels, side_pixels, device=device)
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

    def get_v(x_curr, t_val, label_tensor):
        B = x_curr.shape[0]
        t_tensor = torch.full((B,), t_val, device=device, dtype=torch.float32)
        
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

        with torch.amp.autocast('cuda'):
            if label_tensor is not None:
                label_tensor = (label_tensor - imagenet_mean) / imagenet_std
                v = m(x_curr, t_tensor, label_tensor)
            else:
                v = m(x_curr, t_tensor)
        
        x1_pred = torch.clamp(x_curr + (1 - t_val) * v, -1.0, 1.0)
        if t_val < 1.0:
            v = (x1_pred - x_curr) / (1 - t_val)
            
        return v

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
                v1 = get_v(x, t_val, label_tensor)
                x_euler = x + v1 * dt
                
                if i < num_steps - 1:
                    t_next  = (i + 1) / num_steps
                    v2      = get_v(x_euler, t_next, label_tensor)
                    x = x + (v1 + v2) / 2.0 * dt
                else:
                    x = x_euler

                snapshot['v_field'] = v1.cpu().squeeze()
            else:
                snapshot['v_field'] = None

            snapshots.append(snapshot)

    return snapshots