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
        self.base_loss = nn.SmoothL1Loss(reduction='none', beta=0.05) if base_type == "L1" else nn.MSELoss(reduction='none')
        self.grad_weight = grad_weight

    def forward(self, v_pred, v_target, x1_pred, x_target):
        base = self.base_loss(v_pred, v_target).mean(dim=(1, 2, 3))
        grad = compute_gradient_loss(x1_pred, x_target)
        return base + (self.grad_weight * grad)

class PerSampleLoss(nn.Module):
    def __init__(self, base_type="L1"):
        super().__init__()
        self.loss = nn.SmoothL1Loss(reduction='none', beta=0.05) if base_type == "L1" else nn.MSELoss(reduction='none')

    def forward(self, v_pred, v_target, x1_pred=None, x_target=None):
        return self.loss(v_pred, v_target).mean(dim=(1, 2, 3))
    
LOSS_MAP = {
    "L1":       PerSampleLoss("L1"),
    "MSE":      PerSampleLoss("MSE"),
    "L1_Grad":  CombinedLoss(base_type="L1",  grad_weight=0.5),
    "MSE_Grad": CombinedLoss(base_type="MSE", grad_weight=0.5),
}

def evaluate(model, dataloader_val, device='cpu', loss_fn_str='L1', weight_type='quad', time_sampling='uniform'):
    if loss_fn_str not in LOSS_MAP.keys():
        raise ValueError(f"Loss function {loss_fn_str} not supported.")

    loss_fn = LOSS_MAP[loss_fn_str]
    model.eval()
    total_loss_val = 0

    with torch.no_grad():
        for x, y in dataloader_val:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            x0 = torch.randn_like(x)
            if time_sampling == "logit_normal":
                t = torch.sigmoid(torch.randn(size=(x.shape[0],), device=device))
            else:
                t = torch.rand(size=(x.shape[0],), device=device)
                
            xt = t[:, None, None, None]*x + (1-t[:, None, None, None])*x0
            v = x-x0

            with torch.amp.autocast('cuda'):
                v_pred = model(xt, t, y).view_as(v)
                x1_pred = xt + (1 - t[:, None, None, None]) * v_pred
                per_sample_loss = loss_fn(v_pred, v, x1_pred, x)
                
            weights = get_loss_weights(t, weight_type=weight_type)
            loss = (per_sample_loss * weights).mean()
            total_loss_val += loss.item()

    model.train()
    return total_loss_val / len(dataloader_val)


def train(model, optimizer, epochs, scheduler, dataloader_train, device='cpu', loss_fn_str='L1_Grad', dataloader_val=None,
          overfit_x0=None, weight_type='quad', side_pixels=128, patience=5, time_sampling='uniform', eval_freq=10):
    
    _GPU_SPATIAL = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomResizedCrop(size=(side_pixels, side_pixels), scale=(0.75, 1.0), antialias=True),
    ])
    _GPU_COLOR = v2.ColorJitter(brightness=0.15, contrast=0.1, saturation=0.1)

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

    best_val   = float('inf')
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0

        for x, y in dataloader_train:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

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
                if time_sampling == "logit_normal":
                    t = torch.sigmoid(torch.randn(size=(x.shape[0],), device=device))
                else:
                    t = torch.rand(size=(x.shape[0],), device=device)
                    
                xt = t[:, None, None, None] * x + (1 - t[:, None, None, None]) * x0
                v = x - x0

                v_pred = model(xt, t, y).view_as(v)
                x1_pred = xt + (1 - t[:, None, None, None]) * v_pred
                
                per_sample_loss = loss_fn(v_pred, v, x1_pred, x)
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
        current_lr = scheduler.get_last_lr()[0] if scheduler and is_batch_scheduler else optimizer.param_groups[0]['lr']

        if mlflow.active_run():
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            mlflow.log_metric("learning_rate", current_lr, step=epoch)

        if dataloader_val is not None and (epoch % eval_freq == 0 or epoch == 1):
            avg_loss_val = evaluate(model, dataloader_val, device, loss_fn_str, weight_type, time_sampling)

            if mlflow.active_run():
                mlflow.log_metric("val_loss", avg_loss_val, step=epoch)

            if avg_loss_val < best_val:
                best_val = avg_loss_val
                no_improve = 0
                print(f'Epoch {epoch:04d}/{epochs} | Train: {avg_loss:.4f} | Val: {avg_loss_val:.4f} | LR: {current_lr:.6f} *** NEW BEST! ***', flush=True)
                
                m = model.module if hasattr(model, 'module') else model
                torch.save(m.state_dict(), "model_best.pth")
                if mlflow.active_run():
                    mlflow.log_metric("best_val_loss", best_val, step=epoch)
            else:
                no_improve += 1
                print(f'Epoch {epoch:04d}/{epochs} | Train: {avg_loss:.4f} | Val: {avg_loss_val:.4f} | LR: {current_lr:.6f} | Patience: {no_improve}/{patience}', flush=True)
                
                if no_improve >= patience:
                    print(f"\nEarly stopping triggered at epoch {epoch}!", flush=True)
                    break
            
            if scheduler is not None and not is_batch_scheduler:
                if is_plateau_scheduler:
                    scheduler.step(avg_loss_val)
                else:
                    scheduler.step()

        elif dataloader_val is None:
            print(f'Epoch {epoch:04d}/{epochs} | Train: {avg_loss:.4f} | LR: {current_lr:.6f}', flush=True)
            
            if scheduler is not None and not is_batch_scheduler:
                if is_plateau_scheduler:
                    scheduler.step(avg_loss)
                else:
                    scheduler.step()