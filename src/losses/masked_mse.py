class MaskedMSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, mask):
        pred = pred[:, 0]
        target = target[:, 0]
        mask = mask[:, 0].bool()
        se = (pred - target) ** 2
        return se[mask].mean() if mask.any() else se.mean()