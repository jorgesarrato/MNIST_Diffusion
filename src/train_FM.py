import torch
import torch.optim as optim
import torch.nn as nn


def evaluate(model, dataloader_val, device='cpu'):
    model.eval()

    total_loss_val = 0

    for x, y in dataloader_val:
        x = x.to(device)

        x0 = torch.randn_like(x)
        t = torch.rand(size=(x.shape[0],), device=device)
        xt = t[:, None, None, None]*x + (1-t[:, None, None, None])*x0
        v = x-x0

        v_pred = model(xt, t).view_as(v)

        loss = nn.MSELoss()(v, v_pred)

        total_loss_val += loss.item()

    return total_loss_val/len(dataloader_val)


def train(model, optimizer, epochs, scheduler, dataloader_train, device='cpu', dataloader_val = None):
    model.to(device)

    for epoch in range(epochs):
        total_loss = 0

        model.train()
        for x, y in dataloader_train:
            x = x.to(device)

            optimizer.zero_grad()
            
            x0 = torch.randn_like(x)
            t = torch.rand(size=(x.shape[0],), device=device)
            xt = t[:, None, None, None]*x + (1-t[:, None, None, None])*x0
            v = x-x0

            v_pred = model(xt, t).view_as(v)

            loss = nn.MSELoss()(v, v_pred)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss/len(dataloader_train)

        if dataloader_val is not None:
            avg_loss_val = evaluate()

            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, Val_Loss: {avg_loss_val:.6f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
            scheduler.step(avg_loss_val)

        else:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
            scheduler.step(avg_loss)
        
