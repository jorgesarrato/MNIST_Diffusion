import torch
import torch.nn as nn
import mlflow
from torchvision.transforms import v2
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
import torch.distributed as dist

from utils.losses import FlowMatchingLoss, compute_gradient_loss, ScaleInvariantLoss

def sync_ema_buffers(base_model, ema_model):
    trained_base = base_model.module if isinstance(base_model, nn.DataParallel) or isinstance(base_model, nn.parallel.DistributedDataParallel) else base_model
    if hasattr(ema_model, 'module') and (isinstance(ema_model.module, nn.DataParallel) or isinstance(ema_model.module, nn.parallel.DistributedDataParallel)):
        ema_base = ema_model.module.module
    elif hasattr(ema_model, 'module'):
        ema_base = ema_model.module
    else:
        ema_base = ema_model
        
    for ema_buf, train_buf in zip(ema_base.buffers(), trained_base.buffers()):
        ema_buf.data.copy_(train_buf.data)

def get_loss_weights(t, weight_type="quad", min_weight=0.1):
    if weight_type == "quad":
        return (t**2) + min_weight
    else:
        return torch.ones_like(t)

def evaluate(model, loss_fn, dataloader_val, device='cpu', weight_type='quad', time_sampling='uniform'):
    model.eval()
    
    metrics = {"val_total": 0.0, "val_l1": 0.0, "val_grad": 0.0, "val_si": 0.0}
    
    l1_tracker = nn.SmoothL1Loss(reduction='none', beta=0.05)
    si_tracker = ScaleInvariantLoss()

    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    with torch.no_grad():
        for x, y, mask in dataloader_val:
            x, y, mask = x.to(device, non_blocking=True), y.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            y = (y - imagenet_mean) / imagenet_std

            x0 = torch.randn_like(x)
            if time_sampling == "logit_normal":
                t = torch.sigmoid(torch.randn(size=(x.shape[0],), device=device))
            else:
                t = torch.rand(size=(x.shape[0],), device=device)
                
            xt = t[:, None, None, None]*x + (1-t[:, None, None, None])*x0
            v = x - x0

            with torch.amp.autocast('cuda'):
                v_pred = model(xt, t, y).view_as(v)
                x1_pred = xt + (1 - t[:, None, None, None]) * v_pred
                                
                per_sample_loss = loss_fn(v_pred, v, x1_pred, x, y, mask)
                weights = get_loss_weights(t, weight_type=weight_type)
                metrics["val_total"] += (per_sample_loss * weights).mean().item()
                
                l1_raw = l1_tracker(v_pred, v) * mask
                valid_pixels = torch.clamp(mask.sum(dim=(1, 2, 3)), min=1e-6)
                metrics["val_l1"] += (l1_raw.sum(dim=(1, 2, 3)) / valid_pixels).mean().item()
                
                metrics["val_grad"] += compute_gradient_loss(x1_pred, x, mask).mean().item()
                metrics["val_si"] += si_tracker(x1_pred, x, mask).mean().item()

    model.train()
    n = len(dataloader_val)
    return {k: v / n for k, v in metrics.items()}


def train(model, optimizer, epochs, scheduler, dataloader_train, device='cpu', 
          loss_type='L1', grad_weight=0.5, si_weight=0.0, edge_weight=0.0, 
          dataloader_val=None, overfit_x0=None, weight_type='quad', 
          side_pixels=128, patience=5, time_sampling='uniform', eval_freq=10, 
          ema_decay=0.999, cond_drop_prob=0.20,
          is_distributed=False, train_sampler=None, is_main_process=True, log_depth=False,
          start_epoch=1, best_val=float('inf'), ema_state=None):
    
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    _GPU_SPATIAL = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        #v2.RandomRotation(degrees=5),
    ])
    _GPU_COLOR = v2.ColorJitter(brightness=0.25, contrast=0.2, saturation=0.2, hue=0.05)
    _GPU_EXTRA = v2.Compose([
        v2.RandomGrayscale(p=0.05),
        v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    ])

    loss_fn = FlowMatchingLoss(base_type=loss_type, grad_weight=grad_weight, si_weight=si_weight, edge_weight=edge_weight, log_depth=log_depth)
    
    ema_avg_fn = get_ema_multi_avg_fn(ema_decay)
    ema_model = AveragedModel(model, multi_avg_fn=ema_avg_fn).to(device)

    if ema_state is not None:
            ema_model.load_state_dict(ema_state)

    use_cuda = 'cuda' in str(device)
    device_type = 'cuda' if use_cuda else 'cpu'
    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda)

    is_batch_scheduler = isinstance(scheduler, (torch.optim.lr_scheduler.OneCycleLR, torch.optim.lr_scheduler.CyclicLR, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts))
    is_plateau_scheduler = isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

    no_improve = 0

    for epoch in range(start_epoch, epochs + 1):
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        total_loss = 0

        for x, y, mask in dataloader_train:
            x, y, mask = x.to(device, non_blocking=True), y.to(device, non_blocking=True), mask.to(device, non_blocking=True)

            stacked = torch.cat([y, x, mask], dim=1)
            stacked = _GPU_SPATIAL(stacked)
            y, x, mask = stacked[:, :3], stacked[:, 3:4], stacked[:, 4:5]
            y = _GPU_COLOR(y)
            y = _GPU_EXTRA(y)

            y = (y - imagenet_mean) / imagenet_std

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

                drop_mask = torch.rand(x.shape[0], device=device) < cond_drop_prob
                v_pred = model(xt, t, y, drop_mask=drop_mask).view_as(v)

                x1_pred = xt + (1 - t[:, None, None, None]) * v_pred
                
                # Pass 'y' (the RGB image) to satisfy the edge-aware loss
                per_sample_loss = loss_fn(v_pred, v, x1_pred, x, y, mask)
                weights = get_loss_weights(t, weight_type=weight_type)
                loss = (per_sample_loss * weights).mean()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            ema_model.update_parameters(model)
            
            if scheduler is not None and is_batch_scheduler:
                scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader_train)
        
        if is_distributed:
            avg_loss_tensor = torch.tensor(avg_loss, device=device)
            dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = (avg_loss_tensor / dist.get_world_size()).item()
            
        current_lr = scheduler.get_last_lr()[0] if scheduler and is_batch_scheduler else optimizer.param_groups[0]['lr']

        if is_main_process and mlflow.active_run():
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            mlflow.log_metric("learning_rate", current_lr, step=epoch)

        if dataloader_val is not None and (epoch % eval_freq == 0 or epoch == 1):
            sync_ema_buffers(model, ema_model)
            val_metrics = evaluate(ema_model, loss_fn, dataloader_val, device, weight_type, time_sampling)
            
            if is_distributed:
                metrics_tensor = torch.tensor([
                    val_metrics["val_total"], val_metrics["val_l1"], 
                    val_metrics["val_grad"], val_metrics["val_si"]
                ], device=device)
                
                dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
                metrics_tensor /= dist.get_world_size()
                
                val_metrics["val_total"] = metrics_tensor[0].item()
                val_metrics["val_l1"] = metrics_tensor[1].item()
                val_metrics["val_grad"] = metrics_tensor[2].item()
                val_metrics["val_si"] = metrics_tensor[3].item()

            if is_main_process and mlflow.active_run():
                mlflow.log_metrics(val_metrics, step=epoch)

            if val_metrics["val_total"] < best_val:
                best_val = val_metrics["val_total"]
                no_improve = 0
                
                if is_main_process:
                    print(f'Epoch {epoch:04d}/{epochs} | Train: {avg_loss:.4f} | Val Total: {best_val:.4f} | Val L1: {val_metrics["val_l1"]:.4f} | LR: {current_lr:.6f} *** NEW BEST! ***', flush=True)
                    ckpt = {
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict() if is_distributed else model.state_dict(),
                        'ema_state_dict': ema_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                        'best_val': best_val,
                        'mlflow_run_id': mlflow.active_run().info.run_id if mlflow.active_run() else None
                    }
                    torch.save(ckpt, "checkpoint_latest.pth")
                    print(f"*** Full checkpoint saved at epoch {epoch} ***")
                    if mlflow.active_run():
                        mlflow.log_metric("best_val_loss", best_val, step=epoch)
            else:
                no_improve += 1
                if is_main_process:
                    print(f'Epoch {epoch:04d}/{epochs} | Train: {avg_loss:.4f} | Val Total: {val_metrics["val_total"]:.4f} | Val L1: {val_metrics["val_l1"]:.4f} | Patience: {no_improve}/{patience}', flush=True)
                
            if no_improve >= patience:
                if is_main_process:
                    print(f"\nEarly stopping triggered at epoch {epoch}!", flush=True)
                break
            
            if scheduler is not None and not is_batch_scheduler:
                if is_plateau_scheduler:
                    scheduler.step(val_metrics["val_total"])
                else:
                    scheduler.step()

        elif dataloader_val is None:
            if is_main_process:
                print(f'Epoch {epoch:04d}/{epochs} | Train: {avg_loss:.4f} | LR: {current_lr:.6f}', flush=True)
            
            if scheduler is not None and not is_batch_scheduler:
                if is_plateau_scheduler:
                    scheduler.step(avg_loss)
                else:
                    scheduler.step()