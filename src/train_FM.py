import torch
import torch.nn as nn
import mlflow

LOSS_MAP = {
    "L1": nn.L1Loss(),
    "MSE": nn.MSELoss()
}


def evaluate(model, dataloader_val, device='cpu', loss_fn_str='L1'):
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

            loss = loss_fn(v, v_pred)

            total_loss_val += loss.item()

    return total_loss_val/len(dataloader_val)


def train(model, optimizer, epochs, scheduler, dataloader_train, device='cpu', loss_fn_str = 'L1', dataloader_val = None, overfit_x0=None):
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

            loss = loss_fn(v, v_pred)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss/len(dataloader_train)
        current_lr = optimizer.param_groups[0]['lr']

        if mlflow.active_run():
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            mlflow.log_metric("learning_rate", current_lr, step=epoch)

        if dataloader_val is not None:
            avg_loss_val = evaluate(model, dataloader_val, device, loss_fn_str)

            if mlflow.active_run():
                mlflow.log_metric("val_loss", avg_loss_val, step=epoch)

            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, Val_Loss: {avg_loss_val:.6f}, LR: {current_lr:.6f}')
            scheduler.step(avg_loss_val)

        else:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, LR: {current_lr:.6f}')
            scheduler.step(avg_loss)
        
