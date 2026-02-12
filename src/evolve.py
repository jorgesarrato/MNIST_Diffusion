import torch

def save_flow_evolution(model, x = None, label = None, device = 'cpu', num_steps=50):
    if x is None:
        x = torch.randn(1, 1, 28, 28).to(device)
    else:
        x = x.to(device)

    dt = 1 / num_steps

    model.eval()
    snapshots = []

    with torch.no_grad():
        for i in range(num_steps + 1):
            snapshot = {}
            t_val = i / num_steps

            snapshot['image'] = x.cpu().squeeze()

            if i < num_steps:
                t_tensor = torch.tensor([t_val], device=device)

                if label is not None:
                    label_tensor = torch.tensor([label], device=device, dtype=torch.int32)
                    v = model(x, t_tensor, label_tensor)
                else:
                    v = model(x, t_tensor)
                    
                x = x + v * dt
                snapshot['v_field'] = v.cpu().squeeze()
            else:
                snapshot['v_field'] = None

            snapshot['t'] = t_val
            snapshots.append(snapshot)

    return snapshots

