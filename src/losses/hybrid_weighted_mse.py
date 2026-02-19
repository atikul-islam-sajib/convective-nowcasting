import torch
import torch.nn as nn


class HybridWeightedMSE(nn.Module):
    def __init__(
        self,
        eps=1e-3,
        weight_threshold_1=15.0,
        weight_value_1=2.0,
        weight_threshold_2=50.0,
        weight_value_2=5.0,
        weight_threshold_3=100.0,
        weight_value_3=10.0,
    ):
        super().__init__()
        self.eps = eps
        self.weight_threshold_1 = weight_threshold_1
        self.weight_value_1 = weight_value_1
        self.weight_threshold_2 = weight_threshold_2
        self.weight_value_2 = weight_value_2
        self.weight_threshold_3 = weight_threshold_3
        self.weight_value_3 = weight_value_3

    def inverse_transform(self, y):
        return torch.clamp((10.0**y - self.eps) * 12.0, min=0.0, max=400.0)

    def compute_weights(self, target_mmh):
        weights = torch.ones_like(target_mmh)
        weights = torch.where(
            target_mmh > self.weight_threshold_1,
            torch.tensor(
                self.weight_value_1, device=weights.device, dtype=weights.dtype
            ),
            weights,
        )
        weights = torch.where(
            target_mmh > self.weight_threshold_2,
            torch.tensor(
                self.weight_value_2, device=weights.device, dtype=weights.dtype
            ),
            weights,
        )
        weights = torch.where(
            target_mmh > self.weight_threshold_3,
            torch.tensor(
                self.weight_value_3, device=weights.device, dtype=weights.dtype
            ),
            weights,
        )
        return weights

    def forward(self, pred, target, mask):
        pred_log = pred[:, 0]
        target_log = target[:, 0]
        mask = mask[:, 0].bool()

        target_mmh = self.inverse_transform(target_log)
        weights = self.compute_weights(target_mmh)
        se = (pred_log - target_log) ** 2
        loss = weights * se

        return loss[mask].mean() if mask.any() else loss.mean()
