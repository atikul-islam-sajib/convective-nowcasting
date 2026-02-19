import torch
import torch.nn as nn


def compute_ssim(pred, target, mask):
    pred = pred[:, 0]
    target = target[:, 0]
    mask = mask[:, 0].bool()

    if not mask.any():
        return 0.0

    pred_masked = pred[mask]
    target_masked = target[mask]

    mu_pred = pred_masked.mean()
    mu_target = target_masked.mean()
    var_pred = pred_masked.var()
    var_target = target_masked.var()
    cov = ((pred_masked - mu_pred) * (target_masked - mu_target)).mean()

    data_range = 5.6
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim = ((2 * mu_pred * mu_target + C1) * (2 * cov + C2)) / (
        (mu_pred**2 + mu_target**2 + C1) * (var_pred + var_target + C2)
    )

    return ssim.item()


def compute_psnr_mm(pred_log, target_log, mask, eps=1e-3, max_val=400.0):
    pred_log = pred_log[:, 0]
    target_log = target_log[:, 0]
    mask = mask[:, 0].bool()

    if not mask.any():
        return 0.0

    pred_mm = torch.clamp((10.0**pred_log - eps) * 12.0, 0.0, max_val)
    target_mm = torch.clamp((10.0**target_log - eps) * 12.0, 0.0, max_val)

    mse = ((pred_mm[mask] - target_mm[mask]) ** 2).mean()

    if mse == 0:
        return float("inf")

    psnr = 20 * torch.log10(torch.tensor(max_val, device=mse.device) / torch.sqrt(mse))

    return psnr.item()


def compute_metrics(pred, target, mask):
    pred_single = pred[:, 0]
    target_single = target[:, 0]
    mask_single = mask[:, 0].bool()

    if mask_single.any():
        mae = torch.abs(pred_single[mask_single] - target_single[mask_single]).mean()
        rmse = torch.sqrt(
            ((pred_single[mask_single] - target_single[mask_single]) ** 2).mean()
        )
    else:
        mae = torch.abs(pred_single - target_single).mean()
        rmse = torch.sqrt(((pred_single - target_single) ** 2).mean())

    ssim = compute_ssim(pred, target, mask)
    psnr = compute_psnr_mm(pred, target, mask, eps=1e-3, max_val=400.0)

    return {"mae": mae.item(), "rmse": rmse.item(), "ssim": ssim, "psnr": psnr}