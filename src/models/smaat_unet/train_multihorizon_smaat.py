"""
MULTI-HORIZON TRAINING — SmaAt-UNet
=====================================

SmaAt-UNet (Trebing et al. 2021) adapted for multi-horizon radar nowcasting.

Key differences from ConvLSTM version:
    1. No temporal recurrence — treats 12 timesteps as stacked channels [B, 36, H, W]
    2. No wrapper needed — SmaAt_UNet forward() directly compatible
    3. CBAM attention at every encoder scale
    4. Depthwise separable convolutions — faster and lighter than ConvLSTM
    5. All other code (loss, metrics, dataset, viz) IDENTICAL to v2

Architecture:
    n_channels         = 36   (12 timesteps × 3 channels)
    n_classes          = 4    (t+15, t+30, t+45, t+60)
    kernels_per_layer  = 2
    bilinear           = True
    reduction_ratio    = 16
"""

import os
import json
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.amp import autocast, GradScaler
from skimage.morphology import closing, remove_small_objects, disk
from skimage.metrics import structural_similarity as skimage_ssim
from scipy.ndimage import label
from tqdm import tqdm

from utils.config_loader import load_config
from utils.data_policy import DataPolicy
from datasets.satellite_radar_dataset_multihorizon import SatelliteRadarDataset, SoftLogTransform
from utils.time_utils_multihorizon import compute_forecast_horizons

# SmaAt-UNet
from src.models.smaat_unet.SmaAt_UNet import SmaAt_UNet


# ============================================================
# CONFIG
# ============================================================
class SmaAtTrainingConfig:
    """SmaAt-UNet training config."""

    metadata_train = 'metadata/train_sampled_multihorizon.csv'
    metadata_val   = 'metadata/val_sampled_multihorizon.csv'
    config_yml     = 'config/config.yml'

    # Two-threshold strategy
    operational_threshold    = 15.0
    storm_threshold_json     = 'metadata/storm_threshold.json'
    use_extreme_from_data    = False
    manual_extreme_threshold = 80.0

    # Output directories — separate from ConvLSTM and PredRNN
    checkpoint_dir = 'Checkpoints_SmaAt'
    train_viz_dir  = 'TrainViz_SmaAt'
    val_viz_dir    = 'ValViz_SmaAt'
    resume_from    = None

    # ── SmaAt-UNet architecture ────────────────────────────────
    kernels_per_layer = 2
    bilinear          = True
    reduction_ratio   = 16

    # ── Hyperparameters ────────────────────────────────────────
    num_epochs = 50

    batch_size                  = 32   # SmaAt is lighter — can handle larger batch
    gradient_accumulation_steps = 2    # Effective batch = 64
    num_workers                 = 16

    max_lr        = 5e-4
    div_factor    = 25
    learning_rate = max_lr / div_factor   # 2e-5
    min_lr        = 1e-6

    weight_decay = 1e-4
    grad_clip    = 1.0

    scheduler_type   = 'onecycle'
    pct_start        = 0.3
    final_div_factor = 1000
    T_0              = 10
    T_mult           = 2
    scheduler_patience = 3
    scheduler_factor   = 0.5

    use_mixed_precision = True

    patience  = 35
    min_delta = 1e-5

    # Visualization
    num_viz_samples       = 4
    max_batches_to_search = 20
    viz_every_n_epochs    = 1

    # Storm detection
    storm_min_pixels     = 1
    morphology_disk_size = 4
    storm_min_area_km2   = 50.0
    storm_max_area_km2   = 150.0
    pixel_area_km2       = 1.0

    grad_check_every = 10

    # Loss
    loss_type          = 'hybrid_weighted_mae'
    weight_threshold_1 = 15.0
    weight_value_1     = 2.0
    weight_threshold_2 = 50.0
    weight_value_2     = 5.0
    weight_threshold_3 = 100.0
    weight_value_3     = 7.0

    # Warm-up horizon weights — same as ConvLSTM
    horizon_loss_weights = [0.5, 0.7, 1.0, 2.0]

    beta1 = 0.9
    beta2 = 0.999

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================
# HORIZON WEIGHTS WARM-UP — identical to v2
# ============================================================
def get_horizon_weights(epoch, base_weights):
    n = len(base_weights)
    if epoch < 5:
        return [1.0] * n
    elif epoch < 15:
        return [round(0.7 + 0.6 * (i / (n - 1)), 3) for i in range(n)]
    else:
        return base_weights


# ============================================================
# LOSS FUNCTIONS — identical to v2
# ============================================================
class MaskedMSE(nn.Module):
    def forward(self, pred, target, mask):
        mask = mask.bool()
        se   = (pred - target) ** 2
        return se[mask].mean() if mask.any() else se.mean()


class HybridWeightedMSE(nn.Module):
    def __init__(self, eps=1e-3,
                 weight_threshold_1=15.0, weight_value_1=2.0,
                 weight_threshold_2=50.0, weight_value_2=5.0,
                 weight_threshold_3=100.0, weight_value_3=10.0):
        super().__init__()
        self.eps = eps
        self.weight_threshold_1 = weight_threshold_1
        self.weight_value_1     = weight_value_1
        self.weight_threshold_2 = weight_threshold_2
        self.weight_value_2     = weight_value_2
        self.weight_threshold_3 = weight_threshold_3
        self.weight_value_3     = weight_value_3

    def inverse_transform(self, y):
        mmh = torch.clamp((torch.pow(10.0, y.float()) - self.eps) * 12.0, 0.0, 400.0)
        return mmh.to(y.dtype)

    def compute_weights(self, target_mmh):
        weights = torch.ones_like(target_mmh)
        weights = torch.where(target_mmh > self.weight_threshold_1,
                              torch.tensor(self.weight_value_1, device=weights.device, dtype=weights.dtype), weights)
        weights = torch.where(target_mmh > self.weight_threshold_2,
                              torch.tensor(self.weight_value_2, device=weights.device, dtype=weights.dtype), weights)
        weights = torch.where(target_mmh > self.weight_threshold_3,
                              torch.tensor(self.weight_value_3, device=weights.device, dtype=weights.dtype), weights)
        return weights

    def forward(self, pred, target, mask):
        mask       = mask.bool()
        target_mmh = self.inverse_transform(target)
        weights    = self.compute_weights(target_mmh)
        se         = (pred - target) ** 2
        loss       = weights * se
        return loss[mask].mean() if mask.any() else loss.mean()


class HybridWeightedMAE(nn.Module):
    def __init__(self, eps=1e-3,
                 weight_threshold_1=15.0, weight_value_1=2.0,
                 weight_threshold_2=50.0, weight_value_2=5.0,
                 weight_threshold_3=100.0, weight_value_3=10.0):
        super().__init__()
        self.eps = eps
        self.weight_threshold_1 = weight_threshold_1
        self.weight_value_1     = weight_value_1
        self.weight_threshold_2 = weight_threshold_2
        self.weight_value_2     = weight_value_2
        self.weight_threshold_3 = weight_threshold_3
        self.weight_value_3     = weight_value_3

    def inverse_transform(self, y):
        mmh = torch.clamp((torch.pow(10.0, y.float()) - self.eps) * 12.0, 0.0, 400.0)
        return mmh.to(y.dtype)

    def compute_weights(self, target_mmh):
        weights = torch.ones_like(target_mmh)
        weights = torch.where(target_mmh > self.weight_threshold_1,
                              torch.tensor(self.weight_value_1, device=weights.device, dtype=weights.dtype), weights)
        weights = torch.where(target_mmh > self.weight_threshold_2,
                              torch.tensor(self.weight_value_2, device=weights.device, dtype=weights.dtype), weights)
        weights = torch.where(target_mmh > self.weight_threshold_3,
                              torch.tensor(self.weight_value_3, device=weights.device, dtype=weights.dtype), weights)
        return weights

    def forward(self, pred, target, mask):
        mask       = mask.bool()
        target_mmh = self.inverse_transform(target)
        weights    = self.compute_weights(target_mmh)
        ae         = torch.abs(pred - target)
        loss       = weights * ae
        return loss[mask].mean() if mask.any() else loss.mean()


def compute_multihorizon_loss(loss_fn, pred, target, mask, horizon_weights):
    n_horizons = pred.shape[1]
    assert len(horizon_weights) == n_horizons

    total_loss         = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    per_horizon_losses = []
    weights_sum        = sum(horizon_weights)

    for h_idx in range(n_horizons):
        pred_h   = pred[:, h_idx, :, :]
        target_h = target[:, h_idx, :, :]
        mask_h   = mask[:, h_idx, :, :]
        loss_h   = loss_fn(pred_h, target_h, mask_h)
        total_loss += horizon_weights[h_idx] * loss_h
        per_horizon_losses.append(loss_h.item())

    return total_loss / weights_sum, per_horizon_losses


# ============================================================
# METRICS — identical to v2
# ============================================================
def compute_psnr_mmh(pred, target, mask, eps=1e-3, max_val=400.0):
    mask = mask.bool()
    if not mask.any():
        return 0.0
    pred_mm   = torch.clamp((torch.pow(10.0, pred.float())   - eps) * 12.0, 0.0, max_val)
    target_mm = torch.clamp((torch.pow(10.0, target.float()) - eps) * 12.0, 0.0, max_val)
    mse = ((pred_mm[mask] - target_mm[mask]) ** 2).mean()
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(
        torch.tensor(max_val, device=mse.device, dtype=mse.dtype) / torch.sqrt(mse)
    )
    return psnr.item()


def compute_ssim_mmh(pred, target, mask, eps=1e-3, max_val=400.0):
    mask    = mask.bool()
    pred_mm = torch.clamp((torch.pow(10.0, pred.float())   - eps) * 12.0, 0.0, max_val)
    tgt_mm  = torch.clamp((torch.pow(10.0, target.float()) - eps) * 12.0, 0.0, max_val)
    pred_mm = pred_mm.cpu().numpy()
    tgt_mm  = tgt_mm.cpu().numpy()
    mask_np = mask.cpu().numpy()
    scores  = []
    for b in range(pred_mm.shape[0]):
        if mask_np[b].sum() < 100:
            continue
        p = np.nan_to_num(np.where(mask_np[b], pred_mm[b], np.nan), nan=0.0)
        t = np.nan_to_num(np.where(mask_np[b], tgt_mm[b],  np.nan), nan=0.0)
        score = skimage_ssim(p, t, data_range=max_val,
                             gaussian_weights=True, sigma=1.5,
                             use_sample_covariance=False)
        scores.append(score)
    if len(scores) == 0:
        return float('nan')
    return float(np.mean(scores))


def compute_metrics_per_horizon(pred, target, mask, horizons):
    metrics = {}
    for h_idx, h in enumerate(horizons):
        pred_h   = pred[:, h_idx, :, :]
        target_h = target[:, h_idx, :, :]
        mask_h   = mask[:, h_idx, :, :].bool()
        if mask_h.any():
            mae  = torch.abs(pred_h[mask_h] - target_h[mask_h]).mean()
            rmse = torch.sqrt(((pred_h[mask_h] - target_h[mask_h]) ** 2).mean())
        else:
            mae  = torch.abs(pred_h - target_h).mean()
            rmse = torch.sqrt(((pred_h - target_h) ** 2).mean())
        ssim = compute_ssim_mmh(pred_h, target_h, mask_h)
        psnr = compute_psnr_mmh(pred_h, target_h, mask_h)
        metrics[f't{h}'] = {
            'mae': mae.item(), 'rmse': rmse.item(),
            'ssim': ssim,      'psnr': psnr,
        }
    return metrics


# ============================================================
# TRAINING LOOP
# ============================================================
def train_epoch(model, loader, loss_fn, opt, scheduler, device, grad_clip,
                scaler, accumulation_steps, epoch, total_epochs,
                step_per_batch, horizon_weights, horizons):
    model.train()

    total_loss           = 0.0
    horizon_losses_accum = {f't{h}': 0.0 for h in horizons}
    horizon_mae_accum    = {f't{h}': 0.0 for h in horizons}
    n             = 0
    last_h        = horizons[-1]
    total_batches = len(loader)

    opt.zero_grad()

    pbar = tqdm(
        enumerate(loader),
        total=total_batches,
        desc=f"  Train E{epoch:02d}",
        unit="batch",
        dynamic_ncols=True,
        leave=True,
    )

    for batch_idx, (inputs, targets, masks) in pbar:
        inputs  = inputs.to(device)
        targets = targets.to(device)
        masks   = masks.to(device)

        # SmaAt-UNet: input is already [B, 36, H, W] — no reshaping needed
        with autocast('cuda', enabled=(scaler is not None)):
            pred = model(inputs)   # [B, 4, H, W]

            loss, per_horizon_losses = compute_multihorizon_loss(
                loss_fn, pred, targets, masks, horizon_weights
            )
            loss = loss / accumulation_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        is_accum = (batch_idx + 1) % accumulation_steps == 0
        is_last  = (batch_idx + 1) == total_batches

        if is_accum or is_last:
            if scaler is not None:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                opt.step()

            if step_per_batch and scheduler is not None:
                scheduler.step()

            opt.zero_grad()

        with torch.no_grad():
            metrics = compute_metrics_per_horizon(pred, targets, masks, horizons)

        total_loss += loss.item() * accumulation_steps

        for h_idx, h in enumerate(horizons):
            horizon_losses_accum[f't{h}'] += per_horizon_losses[h_idx]
            horizon_mae_accum[f't{h}']    += metrics[f't{h}']['mae']

        n += 1

        current_lr = opt.param_groups[0]['lr']
        pbar.set_postfix({
            'loss': f"{total_loss/n:.4f}",
            f't{last_h}_mae': f"{horizon_mae_accum[f't{last_h}']/n:.4f}",
            'lr': f"{current_lr:.2e}",
        })

    return {
        'loss':           total_loss / n,
        'horizon_losses': {k: v / n for k, v in horizon_losses_accum.items()},
        'horizon_mae':    {k: v / n for k, v in horizon_mae_accum.items()},
    }


# ============================================================
# VALIDATION LOOP
# ============================================================
@torch.no_grad()
def validate(model, loader, loss_fn, device, horizon_weights, horizons):
    model.eval()

    total_loss            = 0.0
    horizon_losses_accum  = {f't{h}': 0.0 for h in horizons}
    horizon_metrics_accum = {
        f't{h}': {'mae': 0.0, 'rmse': 0.0,
                  'ssim': 0.0, 'ssim_count': 0,
                  'psnr': 0.0, 'psnr_count': 0}
        for h in horizons
    }
    n      = 0
    last_h = horizons[-1]

    pbar = tqdm(
        enumerate(loader),
        total=len(loader),
        desc="  Val     ",
        unit="batch",
        dynamic_ncols=True,
        leave=True,
    )

    for batch_idx, (inputs, targets, masks) in pbar:
        inputs  = inputs.to(device)
        targets = targets.to(device)
        masks   = masks.to(device)

        pred = model(inputs)   # [B, 4, H, W]

        loss, per_horizon_losses = compute_multihorizon_loss(
            loss_fn, pred, targets, masks, horizon_weights
        )
        metrics    = compute_metrics_per_horizon(pred, targets, masks, horizons)
        total_loss += loss.item()

        for h_idx, h in enumerate(horizons):
            horizon_losses_accum[f't{h}']          += per_horizon_losses[h_idx]
            horizon_metrics_accum[f't{h}']['mae']  += metrics[f't{h}']['mae']
            horizon_metrics_accum[f't{h}']['rmse'] += metrics[f't{h}']['rmse']
            psnr_val = metrics[f't{h}']['psnr']
            if math.isfinite(psnr_val):
                horizon_metrics_accum[f't{h}']['psnr']       += psnr_val
                horizon_metrics_accum[f't{h}']['psnr_count'] += 1
            ssim_val = metrics[f't{h}']['ssim']
            if not math.isnan(ssim_val):
                horizon_metrics_accum[f't{h}']['ssim']       += ssim_val
                horizon_metrics_accum[f't{h}']['ssim_count'] += 1

        n += 1

        ssim_val = metrics[f't{last_h}']['ssim']
        ssim_str = f"{ssim_val:.3f}" if not math.isnan(ssim_val) else "nan"
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            f't{last_h}_mae':  f"{metrics[f't{last_h}']['mae']:.4f}",
            f't{last_h}_ssim': ssim_str,
        })

    avg_horizon_metrics = {}
    for h in horizons:
        acc        = horizon_metrics_accum[f't{h}']
        ssim_count = acc['ssim_count']
        psnr_count = acc['psnr_count']
        avg_horizon_metrics[f't{h}'] = {
            'mae':  acc['mae']  / n,
            'rmse': acc['rmse'] / n,
            'ssim': acc['ssim'] / ssim_count if ssim_count > 0 else float('nan'),
            'psnr': acc['psnr'] / psnr_count if psnr_count > 0 else float('nan'),
        }

    return {
        'loss':            total_loss / n,
        'horizon_losses':  {k: v / n for k, v in horizon_losses_accum.items()},
        'horizon_metrics': avg_horizon_metrics,
    }


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    cfg = SmaAtTrainingConfig()

    print("\n" + "="*80)
    print("MULTI-HORIZON TRAINING — SmaAt-UNet")
    print("="*80)
    print(f"Device        : {cfg.device}")
    print(f"Loss Function : {cfg.loss_type}")

    config     = load_config(cfg.config_yml)
    horizons   = compute_forecast_horizons(config.temporal.radar_lead_minutes)
    n_horizons = len(horizons)

    print(f"Horizons      : {horizons}")
    print(f"n_horizons    : {n_horizons}")

    assert len(cfg.horizon_loss_weights) == n_horizons

    # Load thresholds
    operational_thr = cfg.operational_threshold
    extreme_thr     = cfg.manual_extreme_threshold
    print(f"\nOperational threshold : {operational_thr} mm/h")
    print(f"Extreme threshold     : {extreme_thr} mm/h")

    policy = DataPolicy(config.data_policy.nan_handling, mask_nans=config.data_policy.mask_nans)

    print("\n📂 Loading datasets...")
    train_dataset = SatelliteRadarDataset(
        config=config, metadata_csv=cfg.metadata_train,
        mask_nans=policy.mask_nans, validate_files=False, use_cache=False
    )
    val_dataset = SatelliteRadarDataset(
        config=config, metadata_csv=cfg.metadata_val,
        mask_nans=policy.mask_nans, validate_files=False, use_cache=False
    )
    print(f"\n✅ Train: {len(train_dataset):,} | Val: {len(val_dataset):,}")

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=False,
        persistent_workers=True, prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=2
    )

    # Compute input channels from config
    n_timesteps         = config.temporal.history_minutes // config.satellite.cadence_minutes
    n_channels_per_step = len(config.satellite.channels) + 1
    n_input_channels    = n_timesteps * n_channels_per_step   # 36

    print(f"\n🏗️  Building SmaAt-UNet...")
    print(f"   n_input_channels  : {n_input_channels}  ({n_timesteps} timesteps × {n_channels_per_step} ch)")
    print(f"   n_output_channels : {n_horizons}  (one per horizon)")
    print(f"   kernels_per_layer : {cfg.kernels_per_layer}")
    print(f"   reduction_ratio   : {cfg.reduction_ratio}")
    print(f"   bilinear          : {cfg.bilinear}")

    model = SmaAt_UNet(
        n_channels        = n_input_channels,
        n_classes         = n_horizons,
        kernels_per_layer = cfg.kernels_per_layer,
        bilinear          = cfg.bilinear,
        reduction_ratio   = cfg.reduction_ratio,
    ).to(cfg.device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters        : {total_params:,}")

    # Loss
    if cfg.loss_type == 'masked_mse':
        loss_fn = MaskedMSE().to(cfg.device)
        print("✅ Using MaskedMSE")
    elif cfg.loss_type == 'hybrid_weighted_mse':
        loss_fn = HybridWeightedMSE(
            eps=1e-3,
            weight_threshold_1=cfg.weight_threshold_1, weight_value_1=cfg.weight_value_1,
            weight_threshold_2=cfg.weight_threshold_2, weight_value_2=cfg.weight_value_2,
            weight_threshold_3=cfg.weight_threshold_3, weight_value_3=cfg.weight_value_3,
        ).to(cfg.device)
        print("✅ Using HybridWeightedMSE")
    elif cfg.loss_type == 'hybrid_weighted_mae':
        loss_fn = HybridWeightedMAE(
            eps=1e-3,
            weight_threshold_1=cfg.weight_threshold_1, weight_value_1=cfg.weight_value_1,
            weight_threshold_2=cfg.weight_threshold_2, weight_value_2=cfg.weight_value_2,
            weight_threshold_3=cfg.weight_threshold_3, weight_value_3=cfg.weight_value_3,
        ).to(cfg.device)
        print("✅ Using HybridWeightedMAE")
    else:
        raise ValueError(f"Unknown loss_type: {cfg.loss_type}")

    # Optimizer
    opt = optim.AdamW(
        model.parameters(), lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay
    )

    # Scheduler
    step_per_batch  = False
    scheduler       = None

    if cfg.scheduler_type == 'onecycle':
        steps_per_epoch = math.ceil(len(train_loader) / cfg.gradient_accumulation_steps)
        total_steps     = steps_per_epoch * cfg.num_epochs
        scheduler = OneCycleLR(
            opt, max_lr=cfg.max_lr, total_steps=total_steps,
            pct_start=cfg.pct_start, div_factor=cfg.div_factor,
            final_div_factor=cfg.final_div_factor, anneal_strategy='cos'
        )
        step_per_batch = True
        print(f"✅ OneCycleLR: total_steps={total_steps:,}")
    elif cfg.scheduler_type == 'cosine_restart':
        scheduler = CosineAnnealingWarmRestarts(
            opt, T_0=cfg.T_0, T_mult=cfg.T_mult, eta_min=cfg.min_lr
        )
    else:
        scheduler = ReduceLROnPlateau(
            opt, mode='min', factor=cfg.scheduler_factor,
            patience=cfg.scheduler_patience, min_lr=cfg.min_lr
        )

    scaler = GradScaler('cuda') if cfg.use_mixed_precision else None
    if scaler:
        print("✅ Mixed Precision ENABLED")

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.train_viz_dir,  exist_ok=True)
    os.makedirs(cfg.val_viz_dir,    exist_ok=True)

    best_val_loss              = float('inf')
    epochs_without_improvement = 0
    loss_history               = []
    last_h                     = horizons[-1]

    print("\n" + "="*80)
    print("MULTI-HORIZON TRAINING — SmaAt-UNet")
    print("="*80)
    print(f"Model          : SmaAt-UNet (CBAM + Depthwise Separable)")
    print(f"Loss Type      : {cfg.loss_type.upper()}")
    print(f"Horizons       : {horizons} min")
    print(f"Final weights  : {cfg.horizon_loss_weights}  (warm-up first 15 epochs)")
    print(f"Learning Rate  : {cfg.learning_rate:.2e} → {cfg.max_lr:.2e}")
    print(f"Batch Size     : {cfg.batch_size}")
    print(f"Grad Accum     : {cfg.gradient_accumulation_steps}x (effective: {cfg.batch_size * cfg.gradient_accumulation_steps})")
    print(f"Mixed Precision: {cfg.use_mixed_precision}")
    print("="*80 + "\n")

    for epoch in range(1, cfg.num_epochs + 1):
        t0 = time.time()

        print(f"\n{'='*80}")
        print(f"EPOCH {epoch}/{cfg.num_epochs}")
        print(f"{'='*80}")

        current_weights = get_horizon_weights(epoch, cfg.horizon_loss_weights)
        print(f"  Horizon weights: {current_weights}")

        train_metrics = train_epoch(
            model, train_loader, loss_fn, opt, scheduler, cfg.device,
            cfg.grad_clip, scaler, cfg.gradient_accumulation_steps,
            epoch, cfg.num_epochs, step_per_batch,
            current_weights, horizons
        )

        if not step_per_batch and scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(train_metrics['loss'])
            else:
                scheduler.step()

        val_metrics = validate(
            model, val_loader, loss_fn, cfg.device,
            current_weights, horizons
        )

        dt         = time.time() - t0
        current_lr = opt.param_groups[0]['lr']

        print(f"\n  Results (Time: {dt:.1f}s | LR: {current_lr:.6f}):")
        print(f"    Train Loss: {train_metrics['loss']:.6f}")
        print(f"    Val   Loss: {val_metrics['loss']:.6f}")

        print(f"\n  Per-Horizon Validation Metrics:")
        print(f"  {'Horizon':<10} {'Loss':<10} {'MAE':<10} {'RMSE':<10} {'SSIM':<10} {'PSNR(dB)':<12}")
        print(f"  {'-'*62}")
        for h in horizons:
            key    = f't{h}'
            h_loss = val_metrics['horizon_losses'][key]
            h_met  = val_metrics['horizon_metrics'][key]
            marker = " ← PRIMARY" if h == last_h else ""
            ssim_display = h_met['ssim']
            psnr_display = h_met['psnr']
            ssim_fmt = f"{ssim_display:<10.4f}" if not math.isnan(ssim_display) else f"{'nan':<10}"
            psnr_fmt = f"{psnr_display:<12.2f}" if math.isfinite(psnr_display) else f"{'nan':<12}"
            print(f"  t+{h:<7} {h_loss:<10.4f} {h_met['mae']:<10.4f} "
                  f"{h_met['rmse']:<10.4f} {ssim_fmt} {psnr_fmt}{marker}")

        loss_history.append(val_metrics['loss'])

        if len(loss_history) > 1:
            improvement_rate = (loss_history[-2] - loss_history[-1]) / loss_history[-2] * 100
            print(f"\n    Improvement: {improvement_rate:+.2f}% from last epoch")

        # Save last checkpoint
        torch.save({
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'train_loss':           train_metrics['loss'],
            'val_loss':             val_metrics['loss'],
            'val_horizon_metrics':  val_metrics['horizon_metrics'],
            'loss_history':         loss_history,
            'horizons':             horizons,
            'operational_threshold': operational_thr,
            'extreme_threshold':     extreme_thr,
        }, os.path.join(cfg.checkpoint_dir, 'last.pth'))

        # Best model
        if val_metrics['loss'] < best_val_loss - cfg.min_delta:
            improvement   = best_val_loss - val_metrics['loss']
            best_val_loss = val_metrics['loss']
            epochs_without_improvement = 0
            torch.save({
                'epoch':               epoch,
                'model_state_dict':    model.state_dict(),
                'val_loss':            val_metrics['loss'],
                'val_horizon_metrics': val_metrics['horizon_metrics'],
                'horizons':            horizons,
            }, os.path.join(cfg.checkpoint_dir, 'best.pth'))
            print(f"\n  ★★★ NEW BEST MODEL! (improved by {improvement:.6f}) ★★★")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epoch(s)")
            if epochs_without_improvement >= cfg.patience:
                print(f"\n⚡ EARLY STOPPING after {epoch} epochs")
                break

    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE — Best Val Loss: {best_val_loss:.6f}")
    print(f"{'='*80}\n")
