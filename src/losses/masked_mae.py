import torch
import torch.nn as nn


class MaskedMAE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, mask):
        # Remove channel dimension (B,1,H,W) -> (B,H,W)
        pred = pred[:, 0]
        target = target[:, 0]
        mask = mask[:, 0].bool()

        # Absolute error
        ae = torch.abs(pred - target)

        # Apply mask
        return ae[mask].mean() if mask.any() else ae.mean()