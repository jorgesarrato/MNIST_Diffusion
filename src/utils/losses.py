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
    def __init__(self, lam=0.5, depth_min=0.7, depth_max=10.0, log_depth=False):
        super().__init__()
        self.lam = lam
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.log_depth = log_depth

    def forward(self, pred, target, mask):
        eps = 1e-6
        import math

        if self.log_depth:
            log_min = math.log(self.depth_min)
            log_max = math.log(self.depth_max)

            log_pred = (pred + 1) / 2 * (log_max - log_min) + log_min
            log_target = (target + 1) / 2 * (log_max - log_min) + log_min

        else:
            pred_depth = (pred + 1) / 2 * (self.depth_max - self.depth_min) + self.depth_min
            target_depth = (target + 1) / 2 * (self.depth_max - self.depth_min) + self.depth_min

            pred_depth = torch.clamp(pred_depth, min=self.depth_min + eps)
            target_depth = torch.clamp(target_depth, min=self.depth_min + eps)

            log_pred = torch.log(pred_depth)
            log_target = torch.log(target_depth)

        log_diff = (log_pred - log_target) * mask

        n = torch.clamp(mask.sum(dim=(1,2,3)), min=1.0)

        mean = log_diff.sum(dim=(1,2,3)) / n
        term1 = (log_diff ** 2).sum(dim=(1,2,3)) / n
        term2 = mean ** 2

        return term1 - self.lam * term2
    
class FlowMatchingLoss(nn.Module):
    def __init__(
        self,
        base_type="L1",
        grad_weight=0.5,
        si_weight=0.0,
        edge_weight=0.0,
        x1_weight=1.0,
        log_depth=False
    ):
        super().__init__()

        self.base_type = base_type
        self.grad_weight = grad_weight
        self.si_weight = si_weight
        self.edge_weight = edge_weight
        self.x1_weight = x1_weight

        if base_type == "MSE":
            self.base_loss = nn.MSELoss(reduction="none")
        else:
            self.base_loss = nn.SmoothL1Loss(reduction="none", beta=0.1)

        self.si_loss = ScaleInvariantLoss(lam=0.5, log_depth=log_depth)

    def forward(self, v_pred, v_target, x1_pred, x_target, image, mask):
        valid_pixels = torch.clamp(mask.sum(dim=(1, 2, 3)), min=1e-6)

        if self.base_type == "SI":
            base_v = 0.0 
        else:
            base_raw = self.base_loss(v_pred, v_target) * mask
            base_v = base_raw.sum(dim=(1, 2, 3)) / valid_pixels

        if self.x1_weight > 0 and self.base_type != "SI":
            x1_raw = self.base_loss(x1_pred, x_target) * mask
            base_x1 = x1_raw.sum(dim=(1, 2, 3)) / valid_pixels
        else:
            base_x1 = 0.0

        total_loss = base_v + (self.x1_weight * base_x1)

        if self.grad_weight > 0:
            grad_v = compute_gradient_loss(x1_pred, x_target, mask) 
            total_loss = total_loss + self.grad_weight * grad_v

        if self.edge_weight > 0:
            edge = compute_edge_aware_loss(x1_pred, x_target, image, mask)
            total_loss = total_loss + self.edge_weight * edge

        if self.base_type != "SI" and self.si_weight > 0:
            si = self.si_loss(x1_pred, x_target, mask)
            total_loss = total_loss + self.si_weight * si

        return total_loss