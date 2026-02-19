import torch

def save_flow_evolution(model, x=None, label=None, device='cpu', num_steps=50, side_pixels=128):
    model.eval()
    
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

    dt = 1 / num_steps
    snapshots = []

    with torch.no_grad():
        for i in range(num_steps + 1):
            snapshot = {}
            t_val = i / num_steps

            snapshot['image'] = x.cpu().squeeze()
            snapshot['t'] = t_val
            
            snapshot['label'] = label 

            if i < num_steps:
                t_tensor = torch.tensor([t_val], device=device).float()

                if label_tensor is not None:
                    v = model(x, t_tensor, label_tensor)
                else:
                    v = model(x, t_tensor)
                
                x = x + v * dt
                snapshot['v_field'] = v.cpu().squeeze()
            else:
                snapshot['v_field'] = None

            snapshots.append(snapshot)

    return snapshots