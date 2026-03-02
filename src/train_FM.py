import torch
import torch.nn as nn
import mlflow

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


def train(model, optimizer, epochs, scheduler, dataloader_train, device='cpu', loss_fn_str = 'L1_Grad', dataloader_val = None, overfit_x0=None, weight_type='quad'):
    if loss_fn_str not in LOSS_MAP.keys():
        raise ValueError(f"Loss function {loss_fn_str} not supported.")

    loss_fn = LOSS_MAP[loss_fn_str]

    model.to(device)

    for epoch in range(epochs):
        total_loss = 0

        model.train()
        for x, y in dataloader_train:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            
            if overfit_x0 is None:
                x0 = torch.randn_like(x)
            else:
                x0 = overfit_x0.repeat(x.shape[0], 1, 1, 1)


            t = torch.rand(size=(x.shape[0],), device=device)
            xt = t[:, None, None, None]*x + (1-t[:, None, None, None])*x0
            xt.to(device)
            v = x-x0

            v_pred = model(xt, t, y).view_as(v)

            per_sample_loss = loss_fn(v_pred, v)

            weights = get_loss_weights(t, weight_type=weight_type)

            loss = (per_sample_loss * weights).mean()

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss/len(dataloader_train)
        current_lr = optimizer.param_groups[0]['lr']

        if mlflow.active_run():
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            mlflow.log_metric("learning_rate", current_lr, step=epoch)

        if dataloader_val is not None:
            avg_loss_val = evaluate(model, dataloader_val, device, loss_fn_str, weight_type)

            if mlflow.active_run():
                mlflow.log_metric("val_loss", avg_loss_val, step=epoch)

            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, Val_Loss: {avg_loss_val:.6f}, LR: {current_lr:.6f}')
            scheduler.step(avg_loss_val)

        else:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, LR: {current_lr:.6f}')
            scheduler.step(avg_loss)
        
