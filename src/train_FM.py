import torch
import torch.optim as optim
import torch.nn as nn

def train(model, optimizer, epochs, scheduler, dataloader_train , device='cpu'):
    model.train()
    model.to(device)

    for epoch in range(epochs):
        total_loss = 0

        for x, y in dataloader_train:
            x = x.to(device)

            optimizer.zero_grad()
            
            x0 = torch.randn_like(x)
            t = torch.rand(size = [1])
            xt = t[:, None, None, None]*x0 + (1-t[:, None, None, None])*x
            v = xt-x0

            v_pred = model(xt, t)

            loss = nn.MSELoss()(v, v_pred)

            loss.backward()
            optimizer.step()
            scheduler.step(loss.item())

            total_loss+=loss.item()

        avg_loss = total_loss/len(dataloader_train)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}')
        

            