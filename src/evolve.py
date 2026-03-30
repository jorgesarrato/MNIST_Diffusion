import torch
import torch.nn.functional as F

def save_flow_evolution(model, x=None, label=None, device='cpu', num_steps=50, side_pixels=128, guidance_scale=1.5):
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
                
                if guidance_scale > 1.0:
                    mask_keep = torch.zeros(B, dtype=torch.bool, device=device)
                    mask_drop = torch.ones(B, dtype=torch.bool, device=device)
                    
                    v_cond = m(x_curr, t_tensor, label_tensor, drop_mask=mask_keep)
                    v_uncond = m(x_curr, t_tensor, label_tensor, drop_mask=mask_drop)
                    
                    v = v_uncond + guidance_scale * (v_cond - v_uncond)
                else:
                    v = m(x_curr, t_tensor, label_tensor)
            else:
                v = m(x_curr, t_tensor)
            
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

def save_full_flow_evolution(model, label, device='cpu', num_steps=50, patch_size=128, stride=64, guidance_scale=1.5, max_batch_size=16):
    """
    Generates the flow evolution for a full-resolution image using sliding window patches.
    """
    m = model.module if hasattr(model, 'module') else model
    m.eval()

    if label.ndim == 3:
        label = label.unsqueeze(0)
    label = label.to(device, dtype=torch.float32)
    
    _, c, h, w = label.shape
    
    pad_h = (stride - (h - patch_size) % stride) % stride
    pad_w = (stride - (w - patch_size) % stride) % stride
    label_padded = F.pad(label, (0, pad_w, 0, pad_h), mode='reflect')
    
    _, _, h_pad, w_pad = label_padded.shape

    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    label_padded = (label_padded - imagenet_mean) / imagenet_std

    coords = []
    for y in range(0, h_pad - patch_size + 1, stride):
        for x in range(0, w_pad - patch_size + 1, stride):
            coords.append((y, x))
            
    w1d = torch.cat([torch.linspace(0.1, 1.0, patch_size // 2), torch.linspace(1.0, 0.1, patch_size // 2)])
    blend_window = (w1d.unsqueeze(0) * w1d.unsqueeze(1)).view(1, 1, patch_size, patch_size).to(device)

    def extract_patches(full_tensor):
        patches = []
        for y, x in coords:
            patches.append(full_tensor[:, :, y:y+patch_size, x:x+patch_size])
        return torch.cat(patches, dim=0)

    def get_v_batched(x_patches, t_val, label_patches):
        B = x_patches.shape[0]
        v_out = torch.zeros_like(x_patches)
        t_tensor = torch.full((B,), t_val, device=device, dtype=torch.float32)
        
        with torch.no_grad(), torch.amp.autocast('cuda'):
            for i in range(0, B, max_batch_size):
                end = min(i + max_batch_size, B)
                x_chunk = x_patches[i:end]
                t_chunk = t_tensor[i:end]
                l_chunk = label_patches[i:end]
                
                if guidance_scale > 1.0:
                    mask_keep = torch.zeros(end - i, dtype=torch.bool, device=device)
                    mask_drop = torch.ones(end - i, dtype=torch.bool, device=device)
                    
                    v_cond = m(x_chunk, t_chunk, l_chunk, drop_mask=mask_keep)
                    v_uncond = m(x_chunk, t_chunk, l_chunk, drop_mask=mask_drop)
                    v_chunk = v_uncond + guidance_scale * (v_cond - v_uncond)
                else:
                    v_chunk = m(x_chunk, t_chunk, l_chunk)
                    
                v_out[i:end] = v_chunk
        return v_out

    def stitch_patches(patches):
        full = torch.zeros(1, 1, h_pad, w_pad, device=device)
        counts = torch.zeros(1, 1, h_pad, w_pad, device=device)
        for i, (y, x) in enumerate(coords):
            full[:, :, y:y+patch_size, x:x+patch_size] += patches[i:i+1] * blend_window
            counts[:, :, y:y+patch_size, x:x+patch_size] += blend_window
        return full / torch.clamp(counts, min=1e-6)

    x_full = torch.randn(1, 1, h_pad, w_pad, device=device)
    l_patches = extract_patches(label_padded)
    
    dt = 1.0 / num_steps
    snapshots = []

    for i in range(num_steps + 1):
        snapshot = {}
        t_val = i / num_steps

        snapshot['image'] = x_full[:, :, :h, :w].cpu().squeeze()
        snapshot['t']     = t_val
        snapshot['label'] = label.cpu().squeeze()

        if i < num_steps:
            x_patches = extract_patches(x_full)
            v1_patches = get_v_batched(x_patches, t_val, l_patches)
            v1_full = stitch_patches(v1_patches)
            
            x_euler_full = x_full + v1_full * dt
            
            if i < num_steps - 1:
                t_next = (i + 1) / num_steps
                x_euler_patches = extract_patches(x_euler_full)
                v2_patches = get_v_batched(x_euler_patches, t_next, l_patches)
                v2_full = stitch_patches(v2_patches)
                
                x_full = x_full + (v1_full + v2_full) / 2.0 * dt
            else:
                x_full = x_euler_full

            snapshot['v_field'] = v1_full[:, :, :h, :w].cpu().squeeze()
        else:
            snapshot['v_field'] = None

        snapshots.append(snapshot)

    return snapshots

def generate_fast_samples(model, label, device='cpu', num_steps=50, patch_size=128, stride=64, guidance_scale=1.5, max_batch_size=16, num_samples=100):
    m = model.module if hasattr(model, 'module') else model
    m.eval()

    if label.ndim == 3:
        label = label.unsqueeze(0)
    label = label.to(device, dtype=torch.float32)
    
    _, c, h, w = label.shape
    pad_h = (stride - (h - patch_size) % stride) % stride
    pad_w = (stride - (w - patch_size) % stride) % stride
    label_padded = F.pad(label, (0, pad_w, 0, pad_h), mode='reflect')
    
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    label_padded = (label_padded - imagenet_mean) / imagenet_std

    coords = [(y, x) for y in range(0, label_padded.shape[2] - patch_size + 1, stride) 
                     for x in range(0, label_padded.shape[3] - patch_size + 1, stride)]
                     
    w1d = torch.cat([torch.linspace(0.1, 1.0, patch_size // 2), torch.linspace(1.0, 0.1, patch_size // 2)])
    blend_window = (w1d.unsqueeze(0) * w1d.unsqueeze(1)).view(1, 1, patch_size, patch_size).to(device)

    def get_v_batched(x_patches, t_val, label_patches):
        B = x_patches.shape[0]
        v_out = torch.zeros_like(x_patches)
        t_tensor = torch.full((B,), t_val, device=device, dtype=torch.float32)
        
        with torch.no_grad(), torch.amp.autocast('cuda'):
            for i in range(0, B, max_batch_size):
                end = min(i + max_batch_size, B)
                if guidance_scale > 1.0:
                    mask_keep = torch.zeros(end - i, dtype=torch.bool, device=device)
                    mask_drop = torch.ones(end - i, dtype=torch.bool, device=device)
                    v_cond = m(x_patches[i:end], t_tensor[i:end], label_patches[i:end], drop_mask=mask_keep)
                    v_uncond = m(x_patches[i:end], t_tensor[i:end], label_patches[i:end], drop_mask=mask_drop)
                    v_out[i:end] = v_uncond + guidance_scale * (v_cond - v_uncond)
                else:
                    v_out[i:end] = m(x_patches[i:end], t_tensor[i:end], label_patches[i:end])
        return v_out

    l_patches = torch.cat([label_padded[:, :, y:y+patch_size, x:x+patch_size] for y, x in coords], dim=0)
    
    final_samples = []
    dt = 1.0 / num_steps
    
    with torch.no_grad():
        for _ in range(num_samples):
            x_full = torch.randn(1, 1, label_padded.shape[2], label_padded.shape[3], device=device)
            for i in range(num_steps):
                t_val = i / num_steps
                x_patches = torch.cat([x_full[:, :, y:y+patch_size, x:x+patch_size] for y, x in coords], dim=0)
                v1_patches = get_v_batched(x_patches, t_val, l_patches)
                
                v1_full = torch.zeros_like(x_full)
                counts = torch.zeros_like(x_full)
                for idx, (y, x) in enumerate(coords):
                    v1_full[:, :, y:y+patch_size, x:x+patch_size] += v1_patches[idx:idx+1] * blend_window
                    counts[:, :, y:y+patch_size, x:x+patch_size] += blend_window
                v1_full /= torch.clamp(counts, min=1e-6)
                
                x_full = x_full + v1_full * dt
            
            final_samples.append(x_full[:, :, :h, :w].cpu().squeeze())

    return torch.stack(final_samples)