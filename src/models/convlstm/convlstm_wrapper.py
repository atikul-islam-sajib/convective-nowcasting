import torch
import torch.nn as nn


import torch
import torch.nn as nn


class ConvLSTMWrapper(nn.Module):
    def __init__(self, encoder, decoder, num_channels, device='cpu'):
        super().__init__()
        from src.models.convlstm.model import ED
        self.model = ED(encoder, decoder)
        self.device = device
        self.num_channels = num_channels
        self.model.to(device)

    def forward(self, x):
        x = x.to(self.device)

        # x: [B, T*C, H, W]
        B, TC, H, W = x.shape
        assert TC % self.num_channels == 0, \
            f"Input channels ({TC}) not divisible by num_channels ({self.num_channels})"

        T = TC // self.num_channels
        x = x.view(B, T, self.num_channels, H, W)

        out = self.model(x)        # [B, 1, 1, H, W]
        out = out.squeeze(1)       # [B, 1, H, W]

        return out

