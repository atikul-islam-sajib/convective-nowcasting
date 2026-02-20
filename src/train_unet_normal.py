"""
Normal Training Script - Direct 60-Minute Radar Prediction (UNet).

This script trains a UNet-based model to predict a future radar field
directly at a fixed lead time (e.g., 60 minutes), using a temporal stack
of satellite history and radar history as input.

Key features
------------
- Configuration-driven training via YAML (config/unet_config_normal.yml).
- Automatic checkpointing and optional resume.
- Support for multiple loss functions:
    - MaskedMSE
    - HybridWeightedMSE
    - HybridWeightedMAE
- Mixed precision training (AMP) with gradient scaling (optional).
- Gradient accumulation and gradient clipping.
- Learning-rate schedulers:
    - OneCycleLR
    - CosineAnnealingWarmRestarts
    - ReduceLROnPlateau
- Full validation metrics:
    - MAE, RMSE, SSIM, PSNR
- Storm detection overlays and visualization grids.

Outputs
-------
- Checkpoints:
    - <checkpoint_dir>/last.pth  (latest state)
    - <checkpoint_dir>/best.pth  (best validation loss)
- Visualizations:
    - <val_viz_dir>/epoch_XXX.png

Typical usage
-------------
python train_unet_normal.py
"""

import os
import json
import time
import yaml
import math
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from scipy.ndimage import label
import matplotlib.pyplot as plt
from models.unet.unet_model import UNet
from losses.masked_mse import MaskedMSE
from torch.utils.data import DataLoader
from utils.data_policy import DataPolicy
from utils.config_loader import load_config
from metrics.metrics import compute_metrics
from torch.cuda.amp import autocast, GradScaler
from utils.detect_storms import detect_storms_two_level
from losses.hybrid_weighted_mse import HybridWeightedMSE
from losses.hybrid_weighted_mae import HybridWeightedMAE
from matplotlib.colors import ListedColormap, BoundaryNorm
from datasets.satellite_radar_dataset import SatelliteRadarDataset, SoftLogTransform
from torch.optim.lr_scheduler import (
    OneCycleLR,
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
)


def load_training_config(yaml_path="config/unet_config_normal.yml"):
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


class TrainingConfig:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_two_thresholds(cfg):
    operational_thr = cfg.operational_threshold

    if cfg.use_extreme_from_data and os.path.exists(cfg.storm_threshold_json):
        with open(cfg.storm_threshold_json, "r") as f:
            info = json.load(f)
        threshold_mm_5min = info["global_threshold"]
        extreme_thr = threshold_mm_5min * 12.0
        extreme_source = f"P{info['percentile']} from data"
    else:
        extreme_thr = cfg.manual_extreme_threshold
        extreme_source = "Manual fallback"

    return operational_thr, extreme_thr, extreme_source


def find_batch_with_best_storms(
    loader, transform, operational_thr, device, max_batches=20
):
    best_batch = None
    best_p99 = 0.0

    loader_iter = iter(loader)
    for batch_idx in range(min(max_batches, len(loader))):
        try:
            inputs, targets, masks = next(loader_iter)
        except StopIteration:
            break

        targets_mm_h = transform(targets[:, 0].cpu().numpy())
        masks_np = masks[:, 0].cpu().numpy()
        targets_masked = np.where(masks_np > 0.5, targets_mm_h, np.nan)

        valid_values = targets_masked[np.isfinite(targets_masked)]
        if len(valid_values) > 0:
            p99 = np.percentile(valid_values, 99)
        else:
            p99 = 0.0

        if p99 > best_p99:
            best_batch = (inputs, targets, masks)
            best_p99 = p99

    if best_batch is None:
        best_batch = next(iter(loader))

    return best_batch


def create_comparison_grid_two_level(
    preds, gts, masks, transform, cfg, operational_thr, extreme_thr, num_samples=8
):
    RAIN_LEVELS = [5, 10, 20, 30, 50, 100]
    RAIN_COLORS = ["#66bb6a", "#ffeb3b", "#ff9800", "#f44336", "#b71c1c", "#7f0000"]
    RAIN_CMAP = ListedColormap(RAIN_COLORS)
    RAIN_NORM = BoundaryNorm(RAIN_LEVELS, RAIN_CMAP.N)

    fig, axes = plt.subplots(4, 4, figsize=(20, 20))

    for i in range(min(num_samples, len(preds))):
        pred = preds[i, 0].cpu().numpy()
        gt = gts[i, 0].cpu().numpy()
        mask = masks[i, 0].cpu().numpy()

        pred_mm = transform(pred)
        gt_mm = transform(gt)

        pred_mm_masked = np.where((mask > 0.5) & (pred_mm >= 5.0), pred_mm, np.nan)
        gt_mm_masked = np.where((mask > 0.5) & (gt_mm >= 5.0), gt_mm, np.nan)

        gt_operational, gt_extreme, n_gt_op, n_gt_ext = detect_storms_two_level(
            gt_mm_masked,
            operational_thr,
            extreme_thr,
            cfg.storm_min_pixels,
            cfg.morphology_disk_size,
            cfg.storm_min_area_km2,
            cfg.storm_max_area_km2,
            cfg.pixel_area_km2,
        )

        pred_operational, pred_extreme, n_pred_op, n_pred_ext = detect_storms_two_level(
            pred_mm_masked,
            operational_thr,
            extreme_thr,
            cfg.storm_min_pixels,
            cfg.morphology_disk_size,
            cfg.storm_min_area_km2,
            cfg.storm_max_area_km2,
            cfg.pixel_area_km2,
        )

        valid_mask = mask > 0.5
        mae = (
            np.abs(pred[valid_mask] - gt[valid_mask]).mean()
            if valid_mask.any()
            else 0.0
        )

        valid_gt = gt_mm_masked[np.isfinite(gt_mm_masked)]
        valid_pred = pred_mm_masked[np.isfinite(pred_mm_masked)]
        gt_p99 = np.percentile(valid_gt, 99) if len(valid_gt) else 0.0
        pred_p99 = np.percentile(valid_pred, 99) if len(valid_pred) else 0.0

        gt_op_pct = (
            (gt_operational.sum() / valid_mask.sum() * 100) if valid_mask.sum() else 0.0
        )
        pred_op_pct = (
            (pred_operational.sum() / valid_mask.sum() * 100)
            if valid_mask.sum()
            else 0.0
        )

        row = i // 2
        col_gt = (i % 2) * 2
        col_pred = col_gt + 1

        ax_gt = axes[row, col_gt]
        im_gt = ax_gt.imshow(
            np.ma.masked_invalid(gt_mm_masked),
            cmap=RAIN_CMAP,
            norm=RAIN_NORM,
            interpolation="nearest",
        )
        if n_gt_op > 0:
            overlay = np.zeros((*gt_operational.shape, 4))
            overlay[gt_operational] = [1.0, 0.5, 0.0, 0.4]
            ax_gt.imshow(overlay)
        if n_gt_ext > 0:
            overlay = np.zeros((*gt_extreme.shape, 4))
            overlay[gt_extreme] = [1.0, 0.0, 0.0, 0.8]
            ax_gt.imshow(overlay)
        ax_gt.set_title(
            f"GT | P99={gt_p99:.1f} mm/h\nConv:{n_gt_op} ({gt_op_pct:.1f}%) | Ext:{n_gt_ext}",
            fontsize=10,
            fontweight="bold",
        )
        ax_gt.axis("off")

        ax_pred = axes[row, col_pred]
        im_pred = ax_pred.imshow(
            np.ma.masked_invalid(pred_mm_masked),
            cmap=RAIN_CMAP,
            norm=RAIN_NORM,
            interpolation="nearest",
        )
        if n_pred_op > 0:
            overlay = np.zeros((*pred_operational.shape, 4))
            overlay[pred_operational] = [1.0, 0.5, 0.0, 0.4]
            ax_pred.imshow(overlay)
        if n_pred_ext > 0:
            overlay = np.zeros((*pred_extreme.shape, 4))
            overlay[pred_extreme] = [1.0, 0.0, 0.0, 0.8]
            ax_pred.imshow(overlay)
        ax_pred.set_title(
            f"Pred | P99={pred_p99:.1f} mm/h\nConv:{n_pred_op} ({pred_op_pct:.1f}%) | Ext:{n_pred_ext} | MAE={mae:.3f}",
            fontsize=10,
            fontweight="bold",
        )
        ax_pred.axis("off")

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im_gt, cax=cbar_ax, ticks=RAIN_LEVELS, spacing="proportional")
    cbar.set_label("Rain Rate (mm/h)", fontsize=12, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 0.90, 1])
    return fig


def save_visualizations(
    model, loader, epoch, transform, cfg, operational_thr, extreme_thr, outdir, device
):
    model.eval()

    inputs, targets, masks = find_batch_with_best_storms(
        loader, transform, operational_thr, device, cfg.max_batches_to_search
    )

    inputs = inputs.to(device)
    targets = targets.to(device)
    masks = masks.to(device)

    with torch.no_grad():
        preds = model(inputs)

    fig = create_comparison_grid_two_level(
        preds,
        targets,
        masks,
        transform,
        cfg,
        operational_thr,
        extreme_thr,
        num_samples=cfg.num_viz_samples,
    )

    os.makedirs(outdir, exist_ok=True)
    filepath = os.path.join(outdir, f"epoch_{epoch:03d}.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def train_epoch(
    model,
    loader,
    loss_fn,
    opt,
    scheduler,
    device,
    grad_clip,
    scaler,
    accumulation_steps,
    epoch,
    total_epochs,
    step_per_batch,
):
    model.train()

    total_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    total_ssim = 0.0
    total_psnr = 0.0
    n = 0

    opt.zero_grad()

    pbar = tqdm(loader, desc="  Training", leave=False, ncols=140)
    num_batches = len(loader)

    for batch_idx, (inputs, targets, masks) in enumerate(pbar):
        inputs = inputs.to(device)
        targets = targets.to(device)
        masks = masks.to(device)

        with autocast(enabled=(scaler is not None)):
            pred = model(inputs)
            loss = loss_fn(pred, targets, masks)
            loss = loss / accumulation_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        is_accumulation_step = (batch_idx + 1) % accumulation_steps == 0
        is_last_batch = (batch_idx + 1) == num_batches
        should_update = is_accumulation_step or is_last_batch

        if should_update:
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
            metrics = compute_metrics(pred, targets, masks)

        total_loss += loss.item() * accumulation_steps
        total_mae += metrics["mae"]
        total_rmse += metrics["rmse"]
        total_ssim += metrics["ssim"]
        total_psnr += metrics["psnr"]
        n += 1

        current_lr = opt.param_groups[0]["lr"]
        pbar.set_postfix(
            {
                "loss": f"{total_loss/n:.4f}",
                "mae": f"{total_mae/n:.4f}",
                "ssim": f"{total_ssim/n:.3f}",
                "psnr": f"{total_psnr/n:.1f}",
                "lr": f"{current_lr:.6f}",
            }
        )

    return {
        "loss": total_loss / n,
        "mae": total_mae / n,
        "rmse": total_rmse / n,
        "ssim": total_ssim / n,
        "psnr": total_psnr / n,
    }


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()

    total_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    total_ssim = 0.0
    total_psnr = 0.0
    n = 0

    pbar = tqdm(loader, desc="  Validating", leave=False, ncols=120)

    for inputs, targets, masks in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)
        masks = masks.to(device)

        pred = model(inputs)
        loss = loss_fn(pred, targets, masks)
        metrics = compute_metrics(pred, targets, masks)

        total_loss += loss.item()
        total_mae += metrics["mae"]
        total_rmse += metrics["rmse"]
        total_ssim += metrics["ssim"]
        total_psnr += metrics["psnr"]
        n += 1

        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "mae": f'{metrics["mae"]:.4f}',
                "ssim": f'{metrics["ssim"]:.3f}',
                "psnr": f'{metrics["psnr"]:.1f}',
            }
        )

    return {
        "loss": total_loss / n,
        "mae": total_mae / n,
        "rmse": total_rmse / n,
        "ssim": total_ssim / n,
        "psnr": total_psnr / n,
    }


def train(
    model,
    train_loader,
    val_loader,
    cfg,
    transform,
    operational_thr,
    extreme_thr,
    start_epoch=1,
):
    """
    Full training loop with checkpointing, validation, and visualization.

    Parameters
    ----------
    model : torch.nn.Module
        UNet model to train.
    train_loader : torch.utils.data.DataLoader
        Training DataLoader.
    val_loader : torch.utils.data.DataLoader
        Validation DataLoader.
    cfg : TrainingConfig
        Training configuration object.
    transform : callable
        Inverse transform mapping model-space radar values to mm/h.
    operational_thr : float
        Operational storm threshold in mm/h.
    extreme_thr : float
        Extreme storm threshold in mm/h.
    start_epoch : int, optional
        Epoch index to start from (used for resuming). Default is 1.
    """
    print("\n" + "=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"Model:     UNet (bilinear={cfg.bilinear})")
    print(f"Loss Type: {cfg.loss_type.upper()}")
    if cfg.loss_type in ["hybrid_weighted_mse", "hybrid_weighted_mae"]:
        print(
            f"  Weights: [{cfg.weight_value_1}x @ >{cfg.weight_threshold_1} mm/h, "
            f"{cfg.weight_value_2}x @ >{cfg.weight_threshold_2} mm/h, "
            f"{cfg.weight_value_3}x @ >{cfg.weight_threshold_3} mm/h]"
        )
    print(f"Learning Rate: {cfg.learning_rate} → {cfg.max_lr}")
    print(f"Scheduler: {cfg.scheduler_type.upper()}")
    print("=" * 80 + "\n")

    # Loss function
    if cfg.loss_type == "masked_mse":
        loss_fn = MaskedMSE().to(cfg.device)
    elif cfg.loss_type == "hybrid_weighted_mse":
        loss_fn = HybridWeightedMSE(
            eps=1e-3,
            weight_threshold_1=cfg.weight_threshold_1,
            weight_value_1=cfg.weight_value_1,
            weight_threshold_2=cfg.weight_threshold_2,
            weight_value_2=cfg.weight_value_2,
            weight_threshold_3=cfg.weight_threshold_3,
            weight_value_3=cfg.weight_value_3,
        ).to(cfg.device)
    elif cfg.loss_type == "hybrid_weighted_mae":
        loss_fn = HybridWeightedMAE(
            eps=1e-3,
            weight_threshold_1=cfg.weight_threshold_1,
            weight_value_1=cfg.weight_value_1,
            weight_threshold_2=cfg.weight_threshold_2,
            weight_value_2=cfg.weight_value_2,
            weight_threshold_3=cfg.weight_threshold_3,
            weight_value_3=cfg.weight_value_3,
        ).to(cfg.device)
    else:
        raise ValueError(f"Unknown loss_type: {cfg.loss_type}")

    # Optimizer
    if cfg.scheduler_type == "cosine_restart":
        initial_lr = cfg.max_lr
    else:
        initial_lr = cfg.learning_rate

    opt = optim.AdamW(
        model.parameters(),
        lr=initial_lr,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )

    # Scheduler
    step_per_batch = False
    scheduler = None

    if cfg.scheduler_type == "onecycle":
        steps_per_epoch = math.ceil(
            len(train_loader) / cfg.gradient_accumulation_steps
        )
        total_steps = steps_per_epoch * cfg.num_epochs

        scheduler = OneCycleLR(
            opt,
            max_lr=cfg.max_lr,
            total_steps=total_steps,
            pct_start=cfg.pct_start,
            div_factor=cfg.div_factor,
            final_div_factor=cfg.final_div_factor,
            anneal_strategy="cos",
        )
        step_per_batch = True
        print(
            f" OneCycleLR: {cfg.max_lr/cfg.div_factor:.2e} → {cfg.max_lr:.2e} → {cfg.max_lr/cfg.final_div_factor:.2e}"
        )
    elif cfg.scheduler_type == "cosine_restart":
        scheduler = CosineAnnealingWarmRestarts(
            opt, T_0=cfg.T_0, T_mult=cfg.T_mult, eta_min=cfg.min_lr
        )
        print(f" CosineAnnealingWarmRestarts: {cfg.max_lr:.2e} → {cfg.min_lr:.2e}")
    else:
        scheduler = ReduceLROnPlateau(
            opt,
            mode="min",
            factor=cfg.scheduler_factor,
            patience=cfg.scheduler_patience,
            min_lr=cfg.min_lr,
        )
        print(f" ReduceLROnPlateau")

    # Mixed precision
    scaler = GradScaler() if cfg.use_mixed_precision else None

    # Create directories
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.train_viz_dir, exist_ok=True)
    os.makedirs(cfg.val_viz_dir, exist_ok=True)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    loss_history = []

    # LOAD CHECKPOINT IF RESUMING
    if cfg.resume_from and os.path.exists(cfg.resume_from):
        print(f"\n RESUMING FROM: {cfg.resume_from}")
        checkpoint = torch.load(cfg.resume_from, map_location=cfg.device)

        model.load_state_dict(checkpoint["model_state_dict"])
        opt.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        loss_history = checkpoint.get("loss_history", [])

        print(f" Resumed from epoch {checkpoint['epoch']}")
        print(f"   Best val loss: {best_val_loss:.6f}\n")

    for epoch in range(start_epoch, cfg.num_epochs + 1):
        t0 = time.time()

        print(f"\n{'='*80}")
        print(f"EPOCH {epoch}/{cfg.num_epochs}")
        print(f"{'='*80}")

        # Training
        train_metrics = train_epoch(
            model,
            train_loader,
            loss_fn,
            opt,
            scheduler,
            cfg.device,
            cfg.grad_clip,
            scaler,
            cfg.gradient_accumulation_steps,
            epoch,
            cfg.num_epochs,
            step_per_batch,
        )

        if not step_per_batch and scheduler is not None:
            if cfg.scheduler_type == "reduce_plateau":
                scheduler.step(val_metrics["loss"])  
            else:
                scheduler.step()

        # Validation
        val_metrics = validate(model, val_loader, loss_fn, cfg.device)

        dt = time.time() - t0
        current_lr = opt.param_groups[0]["lr"]

        print(f"\n  Results (Time: {dt:.1f}s | LR: {current_lr:.6f}):")
        print(
            f"    Train Loss: {train_metrics['loss']:.6f} | MAE: {train_metrics['mae']:.4f} | "
            f"SSIM: {train_metrics['ssim']:.4f} | PSNR: {train_metrics['psnr']:.2f} dB"
        )
        print(
            f"    Val   Loss: {val_metrics['loss']:.6f} | MAE: {val_metrics['mae']:.4f} | "
            f"SSIM: {val_metrics['ssim']:.4f} | PSNR: {val_metrics['psnr']:.2f} dB"
        )

        loss_history.append(val_metrics["loss"])

        if len(loss_history) > 1:
            improvement_rate = (
                (loss_history[-2] - loss_history[-1]) / loss_history[-2] * 100
            )
            print(f" Improvement: {improvement_rate:+.2f}% from last epoch")

        # Visualize
        if epoch % cfg.viz_every_n_epochs == 0 or epoch == 1:
            save_visualizations(
                model,
                val_loader,
                epoch,
                transform,
                cfg,
                operational_thr,
                extreme_thr,
                cfg.val_viz_dir,
                cfg.device,
            )

        # Save checkpoint (last)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "val_mae": val_metrics["mae"],
                "val_ssim": val_metrics["ssim"],
                "val_psnr": val_metrics["psnr"],
                "best_val_loss": best_val_loss,
                "loss_history": loss_history,
            },
            os.path.join(cfg.checkpoint_dir, "last.pth"),
        )

        # Check for improvement
        if val_metrics["loss"] < best_val_loss - cfg.min_delta:
            improvement = best_val_loss - val_metrics["loss"]
            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_metrics["loss"],
                    "val_mae": val_metrics["mae"],
                    "val_rmse": val_metrics["rmse"],
                    "val_ssim": val_metrics["ssim"],
                    "val_psnr": val_metrics["psnr"],
                },
                os.path.join(cfg.checkpoint_dir, "best.pth"),
            )

            print(f"\n   NEW BEST MODEL! (improved by {improvement:.6f}) ★★★")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epoch(s)")

            if epochs_without_improvement >= cfg.patience:
                print(f"\n{'='*80}")
                print(f" EARLY STOPPING after {epoch} epochs")
                print(f"{'='*80}")
                break

    print(f"\n{'='*80}")
    print(f" TRAINING COMPLETE - Best Val Loss: {best_val_loss:.6f}")
    print(f"{'='*80}\n")


# MAIN
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print(" NORMAL TRAINING - DIRECT 60MIN PREDICTION (UNet)")
    print("=" * 80)
    print("Loading configuration from: config/unet_config_normal.yml")
    print("=" * 80 + "\n")

    # Load config
    config_dict = load_training_config("config/unet_config_normal.yml")
    cfg = TrainingConfig(config_dict)

    print(f"Device: {cfg.device}")
    print(f"Loss: {cfg.loss_type}")
    print(f"Batch Size: {cfg.batch_size}")
    print(f"Gradient Accumulation: {cfg.gradient_accumulation_steps}x")
    print(f"Effective Batch: {cfg.batch_size * cfg.gradient_accumulation_steps}")

    # Load thresholds
    operational_thr, extreme_thr, extreme_source = load_two_thresholds(cfg)
    print(f"\nThresholds:")
    print(f"  Operational: {operational_thr:.1f} mm/h")
    print(f"  Extreme: {extreme_thr:.1f} mm/h ({extreme_source})")

    # Load main config
    config = load_config(cfg.main_config)
    policy = DataPolicy(
        config.data_policy.nan_handling, mask_nans=config.data_policy.mask_nans
    )

    # Load datasets
    print("\n Loading datasets...")
    train_dataset = SatelliteRadarDataset(
        config=config,
        metadata_csv=cfg.metadata_train,
        mask_nans=policy.mask_nans,
        validate_files=False,
        use_cache=False,
    )

    val_dataset = SatelliteRadarDataset(
        config=config,
        metadata_csv=cfg.metadata_val,
        mask_nans=policy.mask_nans,
        validate_files=False,
        use_cache=False,
    )

    print(f" Train: {len(train_dataset):,} | Val: {len(val_dataset):,}")

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # Get input dimensions
    sample_input, _, _ = train_dataset[0]
    n_channels = sample_input.shape[0]

    print(f"\n  Building UNet Model...")
    print(f"   Input channels: {n_channels}")
    print(f"   Bilinear upsampling: {cfg.bilinear}")

    # Create UNet model
    model = UNet(
        n_channels=n_channels,
        n_classes=1,
        bilinear=cfg.bilinear,
    ).to(cfg.device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,}")

    # Transform
    transform = SoftLogTransform(eps=1e-3, inverse=True)

    # CHECK FOR RESUME
    start_epoch = 1
    if cfg.resume_from:
        if os.path.exists(cfg.resume_from):
            print(f"\n Will resume from: {cfg.resume_from}")
        else:
            print(f"\n  Resume checkpoint not found: {cfg.resume_from}")
            print("   Starting from scratch")
            cfg.resume_from = None
    else:
        # Check for last.pth in checkpoint dir
        last_checkpoint = os.path.join(cfg.checkpoint_dir, "last.pth")
        if os.path.exists(last_checkpoint):
            print(f"\n Found existing checkpoint: {last_checkpoint}")
            response = input("   Resume from this checkpoint? (y/n): ")
            if response.lower() == "y":
                cfg.resume_from = last_checkpoint

    # Train
    train(
        model,
        train_loader,
        val_loader,
        cfg,
        transform,
        operational_thr,
        extreme_thr,
        start_epoch,
    )

    print("\n" + "=" * 80)
    print(" ALL DONE!")
    print("=" * 80)
    print(f"\nCheckpoints saved in: {cfg.checkpoint_dir}/")
    print(f"Visualizations saved in: {cfg.val_viz_dir}/")
    print(f"\nBest model: {cfg.checkpoint_dir}/best.pth")
    print(f"Last checkpoint: {cfg.checkpoint_dir}/last.pth")
    print("=" * 80 + "\n")