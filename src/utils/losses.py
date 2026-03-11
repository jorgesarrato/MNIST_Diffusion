import torch
import torch.nn as nn

def compute_edge_aware_loss(pred_depth, target_depth, image, mask):
    mask_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]
    mask_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]

    dy_pred = pred_depth[:, :, 1:, :] - pred_depth[:, :, :-1, :]
    dy_target = target_depth[:, :, 1:, :] - target_depth[:, :, :-1, :]

    dx_pred = pred_depth[:, :, :, 1:] - pred_depth[:, :, :, :-1]
    dx_target = target_depth[:, :, :, 1:] - target_depth[:, :, :, :-1]

    dy_img = image[:, :, 1:, :] - image[:, :, :-1, :]
    dx_img = image[:, :, :, 1:] - image[:, :, :, :-1]

    weight_y = torch.exp(-torch.mean(torch.abs(dy_img), dim=1, keepdim=True))
    weight_x = torch.exp(-torch.mean(torch.abs(dx_img), dim=1, keepdim=True))

    grad_y = torch.abs(dy_pred - dy_target) * weight_y * mask_y
    grad_x = torch.abs(dx_pred - dx_target) * weight_x * mask_x

    n_y = torch.clamp(mask_y.sum(dim=(1, 2, 3)), min=1e-6)
    n_x = torch.clamp(mask_x.sum(dim=(1, 2, 3)), min=1e-6)

    grad_y = grad_y.sum(dim=(1, 2, 3)) / n_y
    grad_x = grad_x.sum(dim=(1, 2, 3)) / n_x

    return grad_x + grad_y

def compute_gradient_loss(pred, target, mask):
    mask_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]
    mask_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]

    dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dy_target = target[:, :, 1:, :] - target[:, :, :-1, :]
    
    dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dx_target = target[:, :, :, 1:] - target[:, :, :, :-1]

    dy = torch.abs(dy_pred - dy_target) * mask_y
    dx = torch.abs(dx_pred - dx_target) * mask_x
    
    n_y = torch.clamp(mask_y.sum(dim=(1, 2, 3)), min=1e-6)
    n_x = torch.clamp(mask_x.sum(dim=(1, 2, 3)), min=1e-6)

    grad_y = dy.sum(dim=(1, 2, 3)) / n_y
    grad_x = dx.sum(dim=(1, 2, 3)) / n_x
    
    return grad_y + grad_x

class ScaleInvariantLoss(nn.Module):
    def __init__(self, lam=0.85, depth_min=0.7, depth_max=10.0):
        super().__init__()
        self.lam = lam
        self.depth_min = depth_min
        self.depth_max = depth_max

    def forward(self, pred, target, mask):
        pred_real = ((pred + 1.0) / 2.0) * (self.depth_max - self.depth_min) + self.depth_min
        target_real = ((target + 1.0) / 2.0) * (self.depth_max - self.depth_min) + self.depth_min
        
        eps = 1e-6
        pred_real = torch.clamp(pred_real, min=eps)
        target_real = torch.clamp(target_real, min=eps)

        log_diff = torch.log(pred_real) - torch.log(target_real)
        
        log_diff = log_diff * mask
        
        n = torch.clamp(mask.sum(dim=(1, 2, 3)), min=1e-6)

        term1 = (log_diff ** 2).sum(dim=(1, 2, 3)) / n
        term2 = (log_diff.sum(dim=(1, 2, 3)) ** 2) / (n ** 2)

        return term1 - self.lam * term2

class FlowMatchingLoss(nn.Module):

    def __init__(
        self,
        base_type="L1",
        grad_weight=0.5,
        si_weight=0.0,
        edge_weight=0.0
    ):
        super().__init__()

        self.base_type = base_type
        self.grad_weight = grad_weight
        self.si_weight = si_weight
        self.edge_weight = edge_weight

        if base_type == "MSE":
            self.base_loss = nn.MSELoss(reduction="none")
        else:
            self.base_loss = nn.SmoothL1Loss(reduction="none", beta=0.05)

        self.si_loss = ScaleInvariantLoss(lam=0.85)

    def forward(self, v_pred, v_target, x1_pred, x_target, image):

        mask = (x_target > -0.99).float()

        valid_pixels = torch.clamp(mask.sum(dim=(1, 2, 3)), min=1e-6)

        if self.base_type == "SI":
            base = self.si_loss(x1_pred, x_target, mask)
        else:
            base_raw = self.base_loss(v_pred, v_target) * mask
            base = base_raw.sum(dim=(1, 2, 3)) / valid_pixels

        grad = compute_gradient_loss(x1_pred, x_target, mask)
        total_loss = base + self.grad_weight * grad

        if self.edge_weight > 0:
            edge = compute_edge_aware_loss(x1_pred, x_target, image, mask)
            total_loss = total_loss + self.edge_weight * edge

        if self.base_type != "SI" and self.si_weight > 0:
            si = self.si_loss(x1_pred, x_target, mask)
            total_loss = total_loss + self.si_weight * si

        return total_loss