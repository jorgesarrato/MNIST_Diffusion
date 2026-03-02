import torch
import torch.nn as nn
import mlflow
from torchvision.transforms import v2

def get_loss_weights(t, weight_type="quad", min_weight=0.1):
    if weight_type == "quad":
        return (t**2) + min_weight
        
    else:
        return torch.ones_like(t)

def compute_gradient_loss(pred, target):
    dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dy_target = target[:, :, 1:, :] - target[:, :, :-1, :]
    
    dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dx_target = target[:, :, :, 1:] - target[:, :, :, :-1]

    grad_y = torch.abs(dy_pred - dy_target).mean(dim=(1, 2, 3))
    grad_x = torch.abs(dx_pred - dx_target).mean(dim=(1, 2, 3))
    
    return grad_y + grad_x

class CombinedLoss(nn.Module):
    def __init__(self, base_type="L1", grad_weight=0.5):
        super().__init__()
        self.base_loss = nn.L1Loss(reduction='none') if base_type == "L1" else nn.MSELoss(reduction='none')
        self.grad_weight = grad_weight

    def forward(self, pred, target):
        base = self.base_loss(pred, target).mean(dim=(1, 2, 3))
        grad = compute_gradient_loss(pred, target)
        
        return base + (self.grad_weight * grad)

class PerSampleLoss(nn.Module):
    def __init__(self, base_type="L1"):
        super().__init__()
        self.loss = nn.L1Loss(reduction='none') if base_type == "L1" else nn.MSELoss(reduction='none')

    def forward(self, pred, target):
        return self.loss(pred, target).mean(dim=(1, 2, 3))

LOSS_MAP = {
    "L1":       PerSampleLoss("L1"),
    "MSE":      PerSampleLoss("MSE"),
    "L1_Grad":  CombinedLoss(base_type="L1",  grad_weight=0.5),
    "MSE_Grad": CombinedLoss(base_type="MSE", grad_weight=0.5),
}

def evaluate(model, dataloader_val, device='cpu', loss_fn_str='L1', weight_type='quad'):
    if loss_fn_str not in LOSS_MAP.keys():
        raise ValueError(f"Loss function {loss_fn_str} not supported.")

    loss_fn = LOSS_MAP[loss_fn_str]

    model.eval()

    total_loss_val = 0
    with torch.no_grad():
        for x, y in dataloader_val:
            x = x.to(device)
            y = y.to(device)
            y = (y / 255.0) * 2.0 - 1.0

            x0 = torch.randn_like(x)
            t = torch.rand(size=(x.shape[0],), device=device)
            xt = t[:, None, None, None]*x + (1-t[:, None, None, None])*x0
            v = x-x0

            v_pred = model(xt, t, y).view_as(v)

            per_sample_loss = loss_fn(v_pred, v)

            weights = get_loss_weights(t, weight_type=weight_type)

            loss = (per_sample_loss * weights).mean()

            total_loss_val += loss.item()

    return total_loss_val/len(dataloader_val)

def train(model, optimizer, epochs, scheduler, dataloader_train, device='cpu', loss_fn_str='L1_Grad', dataloader_val=None,
          overfit_x0=None, weight_type='quad', side_pixels=128):
    
    _GPU_SPATIAL = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomResizedCrop(size=(side_pixels, side_pixels), scale=(0.8, 1.0), antialias=True),
    ])
    _GPU_COLOR = v2.ColorJitter(brightness=0.2, contrast=0.2)

    if loss_fn_str not in LOSS_MAP.keys():
        raise ValueError(f"Loss function {loss_fn_str} not supported.")

    loss_fn = LOSS_MAP[loss_fn_str]
    model.to(device)

    use_cuda = 'cuda' in str(device)
    device_type = 'cuda' if use_cuda else 'cpu'
    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda)

    is_batch_scheduler = isinstance(scheduler, (
        torch.optim.lr_scheduler.OneCycleLR, 
        torch.optim.lr_scheduler.CyclicLR, 
        torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
    ))
    is_plateau_scheduler = isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x, y in dataloader_train:
            x = x.to(device)
            y = y.to(device)
            y = (y / 255.0) * 2.0 - 1.0

            stacked = torch.cat([y, x], dim=1)
            stacked = _GPU_SPATIAL(stacked)
            y, x = stacked[:, :3], stacked[:, 3:4]
            y = _GPU_COLOR(y)

            optimizer.zero_grad(set_to_none=True)
            
            if overfit_x0 is None:
                x0 = torch.randn_like(x)
            else:
                x0 = overfit_x0.repeat(x.shape[0], 1, 1, 1)

            with torch.amp.autocast(device_type=device_type, enabled=use_cuda):
                t = torch.rand(size=(x.shape[0],), device=device)
                xt = t[:, None, None, None] * x + (1 - t[:, None, None, None]) * x0
                v = x - x0

                v_pred = model(xt, t, y).view_as(v)
                per_sample_loss = loss_fn(v_pred, v)
                weights = get_loss_weights(t, weight_type=weight_type)
                loss = (per_sample_loss * weights).mean()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            scaler.step(optimizer)
            scaler.update()
            
            if scheduler is not None and is_batch_scheduler:
                scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader_train)
        current_lr = optimizer.param_groups[0]['lr']

        if mlflow.active_run():
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            mlflow.log_metric("learning_rate", current_lr, step=epoch)

        if dataloader_val is not None:
            avg_loss_val = evaluate(model, dataloader_val, device, loss_fn_str, weight_type)

            if mlflow.active_run():
                mlflow.log_metric("val_loss", avg_loss_val, step=epoch)

            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, Val_Loss: {avg_loss_val:.6f}, LR: {current_lr:.6f}')
            
            if scheduler is not None and not is_batch_scheduler:
                if is_plateau_scheduler:
                    scheduler.step(avg_loss_val)
                else:
                    scheduler.step()

        else:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, LR: {current_lr:.6f}')
            
            if scheduler is not None and not is_batch_scheduler:
                if is_plateau_scheduler:
                    scheduler.step(avg_loss)
                else:
                    scheduler.step()