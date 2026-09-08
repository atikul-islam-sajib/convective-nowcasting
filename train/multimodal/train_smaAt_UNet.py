import os
import gc
import json
import math
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from skimage.metrics import structural_similarity as skimage_ssim
from skimage.morphology import closing, remove_small_objects, disk
from scipy.ndimage import label
from tqdm import tqdm

import mlflow
import mlflow.pytorch

from utils.config_loader import load_config
from utils.data_policy import DataPolicy

from datasets.satellite_radar_patch_dataset import SatelliteRadarPatchDataset, SoftLogTransform

from utils.time_utils_multihorizon import compute_forecast_horizons

from src.models.smaat_unet.SmaAt_UNet import SmaAt_UNet


CLIP_MAX_MMH           = 128.0
ARTIFACT_THRESHOLD_MMH = 150.0

SOFTLOG_EPS       = 1.0
SOFTLOG_LOG_NORM  = np.log10(CLIP_MAX_MMH + SOFTLOG_EPS)   # log10(129) ≈ 2.1106
SOFTLOG_ZEROVALUE = 0.0

RAIN_LEVELS = [0.1, 1, 2, 5, 10, 15, 20, 30, 40, 60, 100]
RAIN_COLORS = [
    "#1a5c1a", "#22aa22", "#55cc22", "#ffee00", "#ffaa00",
    "#ff6600", "#ff2200", "#cc0000", "#aa0077", "#ff00ff",
]
RAIN_CMAP = ListedColormap(RAIN_COLORS)
RAIN_NORM  = BoundaryNorm(RAIN_LEVELS, RAIN_CMAP.N)
BG_COLOR   = "#888888"

class MultiHorizonTrainingConfig:

    metadata_train = 'metadata_patch/train_patch_256x256_s64_summer_sampled.csv'
    metadata_val   = 'metadata_patch/val_patch_256x256_s64_summer_sampled.csv'
    config_yml     = 'config/config.yml'

    operational_threshold    = 15.0
    storm_threshold_json     = 'metadata/storm_threshold.json'
    use_extreme_from_data    = True
    manual_extreme_threshold = 35.0

    checkpoint_dir = 'CHECKPOINTS_SMAATUNET_PATCH_1KM_SUMMER_2H_2015_to_2024_t15_to_t60_PRE_FINAL'
    train_viz_dir  = 'TRAINVIS_SMAATUNET_PATCH_1KM_SUMMER_2H_2015_to_2024_t15_to_t60_PRE_FINAL'
    val_viz_dir    = 'VALVIZ_SMAATUNET_PATCH_1KM_SUMMER_2H_2015_to_2024_t15_to_t60_PRE_FINAL'

  
    num_epochs = 50

    batch_size                  = 16
    gradient_accumulation_steps = 2
    num_workers                 = 24
    prefetch_factor             = 2

    learning_rate = 1e-4
    weight_decay  = 1e-3
    grad_clip     = 0.5

    use_mixed_precision = True

    patience  = 30
    min_delta = 1e-4

    ets_report_epochs = 1

    lr_scheduler = 'cosine_annealing'

    one_cycle_max_lr    = 3e-4
    one_cycle_pct_start = 0.3

    cosine_eta_min = 5e-6
    cosine_eta_max = 5e-5
    cosine_T_max   = 30
    cosine_T_0     = 30

    plateau_mode      = 'min'
    plateau_factor    = 0.5
    plateau_patience  = 2
    plateau_threshold = 1e-4

    num_viz_samples       = 4
    max_batches_to_search = 30
    viz_every_n_epochs    = 15
    viz_min_rain_mmh      = 10.0

    storm_min_pixels     = 10
    morphology_disk_size = 4
    storm_min_area_km2   = 20.0
    storm_max_area_km2   = 10000.0
    pixel_area_km2       = 1.0

    loss_type = 'hybrid_weighted_mae'

    weight_threshold_1 = 3.0;  weight_value_1 = 3.0
    weight_threshold_2 = 7.0;  weight_value_2 = 7.0
    weight_threshold_3 = 15.0; weight_value_3 = 15.0
    weight_threshold_4 = 25.0; weight_value_4 = 25.0
    weight_threshold_5 = 35.0; weight_value_5 = 35.0

    use_gradient_loss    = True
    gradient_loss_weight = 0.2

    use_perceptual_loss   = False
    perceptual_weight     = 0.1
    perceptual_layers     = ['relu2_2', 'relu3_4']
    perceptual_input_mode = 'repeat'

    horizon_loss_weights = [1.0, 1.5, 2.0, 2.5]

    compute_verification_metrics = True
    ets_threshold_mmh            = 5.0

    compute_extreme_metrics  = True
    extreme_threshold_mmh    = 15.0

    beta1 = 0.9
    beta2 = 0.999

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    resume_from = None
    
    smaat_kernels_per_layer = 2
    smaat_bilinear          = True
    smaat_reduction_ratio   = 16

    # MLflow
    mlflow_experiment   = 'nowcasting_smaatunet_patch_2015_to_2024_t15_to_t60_PRE_FINAL'
    mlflow_run_name     = 'patch256_2H_smaatunet_wmae_gradloss_2015_to_2024_t15_to_t60_PRE_FINAL'
    mlflow_tracking_uri = 'mlruns'


def mlflow_log_config(cfg, horizons, n_train, n_val, total_params):
    mlflow.log_params({
        'metadata_train':              cfg.metadata_train,
        'metadata_val':                cfg.metadata_val,
        'n_train_samples':             n_train,
        'n_val_samples':               n_val,
        'model_backbone':              'SmaAt_UNet',
        'smaat_kernels_per_layer':     cfg.smaat_kernels_per_layer,
        'smaat_bilinear':              cfg.smaat_bilinear,
        'smaat_reduction_ratio':       cfg.smaat_reduction_ratio,
        'total_params':                total_params,
        'horizons':                    str(horizons),
        'n_horizons':                  len(horizons),
        'num_epochs':                  cfg.num_epochs,
        'batch_size':                  cfg.batch_size,
        'learning_rate':               cfg.learning_rate,
        'weight_decay':                cfg.weight_decay,
        'grad_clip':                   cfg.grad_clip,
        'gradient_accumulation_steps': cfg.gradient_accumulation_steps,
        'num_workers':                 cfg.num_workers,
        'use_mixed_precision':         cfg.use_mixed_precision,
        'lr_scheduler':                cfg.lr_scheduler,
        'cosine_T_max':                cfg.cosine_T_max,
        'cosine_eta_min':              cfg.cosine_eta_min,
        'patience':                    cfg.patience,
        'min_delta':                   cfg.min_delta,
        'loss_type':                   cfg.loss_type,
        'horizon_loss_weights':        str(cfg.horizon_loss_weights),
        'weight_threshold_1':          cfg.weight_threshold_1,
        'weight_value_1':              cfg.weight_value_1,
        'weight_threshold_2':          cfg.weight_threshold_2,
        'weight_value_2':              cfg.weight_value_2,
        'weight_threshold_3':          cfg.weight_threshold_3,
        'weight_value_3':              cfg.weight_value_3,
        'weight_threshold_4':          cfg.weight_threshold_4,
        'weight_value_4':              cfg.weight_value_4,
        'weight_threshold_5':          cfg.weight_threshold_5,
        'weight_value_5':              cfg.weight_value_5,
        'use_gradient_loss':           cfg.use_gradient_loss,
        'gradient_loss_weight':        cfg.gradient_loss_weight,
        'use_perceptual_loss':         cfg.use_perceptual_loss,
        'perceptual_weight':           cfg.perceptual_weight,
        'perceptual_layers':           str(cfg.perceptual_layers),
        'ets_threshold_mmh':           cfg.ets_threshold_mmh,
        'extreme_threshold_mmh':       cfg.extreme_threshold_mmh,
        'operational_threshold':       cfg.operational_threshold,
        'manual_extreme_threshold':    cfg.manual_extreme_threshold,
        'softlog_eps':                 SOFTLOG_EPS,
        'softlog_log_norm':            float(SOFTLOG_LOG_NORM),
        'softlog_zerovalue':           SOFTLOG_ZEROVALUE,
        'clip_max_mmh':                CLIP_MAX_MMH,
        'beta1':                       cfg.beta1,
        'beta2':                       cfg.beta2,
    })


def mlflow_log_epoch(epoch, train_metrics, val_metrics, current_lr,
                     dt, horizons, is_best, ets_thr, ext_thr):
    ets_tag = f'op{int(ets_thr)}mmh'
    ext_tag = f'ext{int(ext_thr)}mmh'

    mlflow.log_metrics({
        'train/loss':                    train_metrics['loss'],
        'val/loss':                      val_metrics['loss'],
        f'val/mean_ets_{ets_tag}':       val_metrics['mean_ets'],
        f'val/mean_csi_{ets_tag}':       val_metrics['mean_csi'],
        'train/lr':                      current_lr,
        'epoch_time_s':                  dt,
        'is_best_epoch':                 float(is_best),
    }, step=epoch)

    for h in horizons:
        key = f't{h}'
        mlflow.log_metrics({
            f'train/loss_{key}': train_metrics['horizon_losses'][key],
            f'train/mae_{key}':  train_metrics['horizon_mae'][key],
        }, step=epoch)

    for h in horizons:
        key   = f't{h}'
        h_met = val_metrics['horizon_metrics'][key]
        h_ets = val_metrics['ets_csi_metrics'].get(key, {})
        h_ext = val_metrics['extreme_metrics'].get(key, {})

        metrics_to_log = {
            f'val/mae_{key}':  h_met['mae'],
            f'val/rmse_{key}': h_met['rmse'],
        }
        if not math.isnan(h_met['ssim']):
            metrics_to_log[f'val/ssim_{key}'] = h_met['ssim']
        if math.isfinite(h_met['psnr']):
            metrics_to_log[f'val/psnr_{key}'] = h_met['psnr']
        if h_ets:
            metrics_to_log[f'val/ets_{ets_tag}_{key}'] = h_ets['ets']
            metrics_to_log[f'val/csi_{ets_tag}_{key}'] = h_ets['csi']
        if h_ext:
            metrics_to_log[f'val/csi_{ext_tag}_{key}']      = h_ext['csi']
            metrics_to_log[f'val/baserate_{ext_tag}_{key}'] = h_ext['base_rate']

        mlflow.log_metrics(metrics_to_log, step=epoch)


def get_horizon_weights(epoch, base_weights):
    return base_weights


def should_report_ets(epoch, ets_report_epochs):
    if isinstance(ets_report_epochs, (list, tuple)):
        return int(epoch) in [int(e) for e in ets_report_epochs]
    return int(epoch) % int(ets_report_epochs) == 0


def verify_clip_config(clip_max_mmh):
    eps      = SOFTLOG_EPS
    log_norm = SOFTLOG_LOG_NORM

    def to_log(x):
        return np.log10(x + eps) / log_norm

    print(f"\n{'='*80}")
    print(f"CLIP CONFIGURATION VERIFICATION  [eps=1.0, zerovalue={SOFTLOG_ZEROVALUE}]")
    print(f"{'='*80}")
    print(f"  clip_max : {clip_max_mmh:.2f} mm/h  |  LOG_NORM={log_norm:.4f}")
    for mmh, label in [(0, '0'), (3, 'w1'), (5, 'ETS'), (7, 'w2'),
                       (10, 'ext'), (15, 'w3'), (25, 'w4'), (35, 'w5'), (128, 'max')]:
        print(f"  {mmh:>4} mm/h ({label:<3}) -> {to_log(mmh):.4f}")
    print(f"{'='*80}\n")


def inverse_transform_np(y_log):
    return np.clip(
        10.0 ** (y_log * float(SOFTLOG_LOG_NORM)) - float(SOFTLOG_EPS),
        0.0, float(CLIP_MAX_MMH)
    )


def inverse_transform_torch(y_log):
    return torch.clamp(
        torch.pow(10.0, y_log.float() * float(SOFTLOG_LOG_NORM)) - float(SOFTLOG_EPS),
        0.0, float(CLIP_MAX_MMH)
    ).to(y_log.dtype)


def load_two_thresholds(cfg):
    operational_thr = cfg.operational_threshold

    if cfg.use_extreme_from_data and os.path.exists(cfg.storm_threshold_json):
        with open(cfg.storm_threshold_json, 'r') as f:
            info = json.load(f)
        extreme_thr    = info['global_threshold']
        extreme_source = f"P{info['percentile']} from data"
    else:
        extreme_thr    = cfg.manual_extreme_threshold
        extreme_source = "Manual fallback"

    print(f"\n{'='*80}")
    print(f"TWO-THRESHOLD STRATEGY  [DATA UNIT: mm/h]")
    print(f"{'='*80}")
    print(f"  Operational : {operational_thr:.1f} mm/h")
    print(f"  Extreme     : {extreme_thr:.1f} mm/h ({extreme_source})")
    print(f"{'='*80}\n")

    return operational_thr, extreme_thr, extreme_source


def detect_storms_two_level(
    rain_mm_h, operational_thr, extreme_thr,
    min_pixels=10, disk_size=4,
    min_area_km2=10.0, max_area_km2=100000.0, pixel_area_km2=1.0
):
    valid = np.isfinite(rain_mm_h)
    if valid.sum() == 0:
        empty = np.zeros_like(rain_mm_h, dtype=bool)
        return empty, empty, 0, 0

    mask_op = valid & (rain_mm_h >= operational_thr)
    if mask_op.sum() == 0:
        empty = np.zeros_like(rain_mm_h, dtype=bool)
        return empty, empty, 0, 0

    mask_op           = closing(mask_op, disk(disk_size))
    labeled_op, n_cand = label(mask_op)
    final_op           = np.zeros_like(mask_op, dtype=bool)
    n_op_storms        = 0

    for rid in range(1, n_cand + 1):
        region   = labeled_op == rid
        area_km2 = region.sum() * pixel_area_km2
        if min_area_km2 <= area_km2 <= max_area_km2:
            final_op[region] = True
            n_op_storms += 1

    mask_ext = final_op & (rain_mm_h >= extreme_thr)
    if mask_ext.sum() == 0:
        return final_op, mask_ext, n_op_storms, 0

    mask_ext           = closing(mask_ext, disk(2))
    mask_ext           = remove_small_objects(mask_ext, min_size=10)
    _, n_extreme_cores = label(mask_ext)

    return final_op, mask_ext, n_op_storms, n_extreme_cores

def find_batch_with_best_storms(loader, transform, operational_thr, device,
                                max_batches=20, viz_min_rain_mmh=15.0):
    best_batch  = None
    best_score  = 0.0
    best_inputs_cpu  = None
    best_targets_cpu = None
    best_masks_cpu   = None

    loader_iter = iter(loader)
    for _ in range(min(max_batches, len(loader))):
        try:
            inputs, targets, masks = next(loader_iter)
        except StopIteration:
            break

        targets_np    = targets.cpu().numpy()
        targets_mmh   = inverse_transform_np(targets_np)
        targets_max   = np.nanmax(targets_mmh, axis=1)
        masks_np      = masks.cpu().numpy()
        masks_max     = np.max(masks_np, axis=1)
        targets_masked = np.where(masks_max > 0.5, targets_max, np.nan)
        score = np.nanmax(targets_masked) if np.isfinite(targets_masked).any() else 0.0

        if score > best_score:
            best_inputs_cpu  = inputs.cpu()
            best_targets_cpu = targets.cpu()
            best_masks_cpu   = masks.cpu()
            best_score       = score

        del inputs, targets, masks, targets_np, targets_mmh
        del targets_max, masks_np, masks_max, targets_masked
        torch.cuda.empty_cache()

    if best_inputs_cpu is None:
        best_inputs_cpu, best_targets_cpu, best_masks_cpu = next(iter(loader))
        best_inputs_cpu  = best_inputs_cpu.cpu()
        best_targets_cpu = best_targets_cpu.cpu()
        best_masks_cpu   = best_masks_cpu.cpu()

    targets_np  = best_targets_cpu.numpy()
    targets_mmh = inverse_transform_np(targets_np)
    targets_max = np.nanmax(targets_mmh, axis=1)
    masks_np    = best_masks_cpu.numpy()
    masks_max   = np.max(masks_np, axis=1)

    sample_max = np.array([
        np.nanmax(np.where(masks_max[b] > 0.5, targets_max[b], np.nan))
        if np.isfinite(np.where(masks_max[b] > 0.5, targets_max[b], np.nan)).any()
        else 0.0
        for b in range(targets_np.shape[0])
    ])

    order   = np.argsort(sample_max)[::-1].copy()
    order   = torch.as_tensor(order, dtype=torch.long)
    inputs  = best_inputs_cpu.index_select(0, order)
    targets = best_targets_cpu.index_select(0, order)
    masks   = best_masks_cpu.index_select(0, order)

    del best_inputs_cpu, best_targets_cpu, best_masks_cpu
    del targets_np, targets_mmh, targets_max, masks_np, masks_max

    return inputs, targets, masks

def compute_ets_csi_from_totals(H, FA, M, CN):
    total     = H + FA + M + CN
    denom_csi = H + FA + M
    csi       = H / denom_csi if denom_csi > 0 else 0.0
    H_random  = (H + FA) * (H + M) / total if total > 0 else 0.0
    denom_ets = H + FA + M - H_random
    ets       = (H - H_random) / denom_ets if denom_ets > 0 else 0.0
    return ets, csi


def accumulate_contingency(pred_mm, gt_mm, mask, threshold_mmh):
    valid    = mask > 0.5
    pred_bin = (pred_mm >= threshold_mmh) & valid
    gt_bin   = (gt_mm   >= threshold_mmh) & valid
    H  = int(( pred_bin &  gt_bin).sum())
    FA = int(( pred_bin & ~gt_bin).sum())
    M  = int((~pred_bin &  gt_bin).sum())
    CN = int((~pred_bin & ~gt_bin & valid).sum())
    return H, FA, M, CN

def save_visualizations(model, loader, epoch, transform, cfg,
                        operational_thr, extreme_thr, horizons, outdir, split_name, device):
    plt.close('all')
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.eval()

    inputs, targets, masks = find_batch_with_best_storms(
        loader, transform, operational_thr, device,
        cfg.max_batches_to_search, cfg.viz_min_rain_mmh
    )

    inputs  = inputs.to(device)
    targets = targets.to(device)
    masks   = masks.to(device)

    with torch.no_grad():
        preds = model(inputs)

    preds_cpu   = preds.cpu()
    targets_cpu = targets.cpu()
    masks_cpu   = masks.cpu()
    del preds, inputs, targets, masks
    torch.cuda.empty_cache()

    n_samples  = min(cfg.num_viz_samples, preds_cpu.shape[0])
    n_horizons = len(horizons)
    n_img_cols = n_horizons * 2
    n_cols     = n_img_cols + 1

    FIG_W = 4.2 * n_img_cols + 1.2
    FIG_H = 4.2 * n_samples

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="#1a1a1a")
    gs  = gridspec.GridSpec(
        n_samples, n_cols, figure=fig,
        width_ratios=[1.0] * n_img_cols + [0.03],
        hspace=0.12, wspace=0.06,
        left=0.02, right=0.97, top=0.95, bottom=0.03
    )
    fig.suptitle(
        f"Epoch {epoch:03d}  |  {split_name.upper()} -- Best Storm Samples  [SmaAt-UNet]",
        color="white", fontsize=13, fontweight="bold", y=0.975
    )

    last_im = None

    for row in range(n_samples):
        for h_idx, h in enumerate(horizons):

            pred_log = preds_cpu[row, h_idx].numpy()
            gt_log   = targets_cpu[row, h_idx].numpy()
            mask_np  = masks_cpu[row, h_idx].numpy()

            pred_mm = inverse_transform_np(pred_log)
            gt_mm   = inverse_transform_np(gt_log)
            valid   = mask_np > 0.5

            DISP_MIN  = 0.1
            pred_disp = np.where(valid & (pred_mm >= DISP_MIN), pred_mm, np.nan)
            gt_disp   = np.where(valid & (gt_mm   >= DISP_MIN), gt_mm,   np.nan)

            gt_op, gt_ext, n_gt_op, n_gt_ext = detect_storms_two_level(
                np.where(valid, gt_mm, np.nan),
                operational_thr, extreme_thr,
                cfg.storm_min_pixels, cfg.morphology_disk_size,
                cfg.storm_min_area_km2, cfg.storm_max_area_km2, cfg.pixel_area_km2)
            pr_op, pr_ext, n_pr_op, n_pr_ext = detect_storms_two_level(
                np.where(valid, pred_mm, np.nan),
                operational_thr, extreme_thr,
                cfg.storm_min_pixels, cfg.morphology_disk_size,
                cfg.storm_min_area_km2, cfg.storm_max_area_km2, cfg.pixel_area_km2)

            mae_mmh  = np.abs(pred_mm[valid] - gt_mm[valid]).mean() if valid.any() else 0.0
            gt_p99   = np.percentile(gt_disp[np.isfinite(gt_disp)], 99) \
                       if np.isfinite(gt_disp).any() else 0.0
            pred_p99 = np.percentile(pred_disp[np.isfinite(pred_disp)], 99) \
                       if np.isfinite(pred_disp).any() else 0.0

            col_gt   = h_idx * 2
            col_pred = h_idx * 2 + 1

            ax_gt = fig.add_subplot(gs[row, col_gt])
            ax_gt.set_facecolor(BG_COLOR)
            last_im = ax_gt.imshow(np.ma.masked_invalid(gt_disp),
                                   cmap=RAIN_CMAP, norm=RAIN_NORM, interpolation="nearest")
            if n_gt_op  > 0:
                ov = np.zeros((*gt_op.shape, 4)); ov[gt_op]  = [1.0, 0.6, 0.0, 0.25]
                ax_gt.imshow(ov, interpolation="nearest")
            if n_gt_ext > 0:
                ov = np.zeros((*gt_ext.shape, 4)); ov[gt_ext] = [1.0, 0.0, 1.0, 0.35]
                ax_gt.imshow(ov, interpolation="nearest")
            ax_gt.axis("off")
            ax_gt.set_title(
                f"GT  t+{h}min\nP99={gt_p99:.1f}  Conv={n_gt_op}  Ext={n_gt_ext}",
                color="white", fontsize=7.5, fontweight="bold", pad=3)
            if h_idx == 0:
                ax_gt.set_ylabel(f"#{row+1}", color="#aaaaaa", fontsize=9,
                                 fontweight="bold", labelpad=4, rotation=0, va="center")
                ax_gt.yaxis.set_label_coords(-0.08, 0.5)

            ax_pred = fig.add_subplot(gs[row, col_pred])
            ax_pred.set_facecolor(BG_COLOR)
            ax_pred.imshow(np.ma.masked_invalid(pred_disp),
                           cmap=RAIN_CMAP, norm=RAIN_NORM, interpolation="nearest")
            if n_pr_op  > 0:
                ov = np.zeros((*pr_op.shape, 4)); ov[pr_op]  = [1.0, 0.6, 0.0, 0.25]
                ax_pred.imshow(ov, interpolation="nearest")
            if n_pr_ext > 0:
                ov = np.zeros((*pr_ext.shape, 4)); ov[pr_ext] = [1.0, 0.0, 1.0, 0.35]
                ax_pred.imshow(ov, interpolation="nearest")
            ax_pred.axis("off")
            ax_pred.set_title(
                f"Pred t+{h}min\nP99={pred_p99:.1f}  Conv={n_pr_op}  MAE={mae_mmh:.1f}",
                color="white", fontsize=7.5, fontweight="bold", pad=3)

    if last_im is not None:
        cbar_ax = fig.add_subplot(gs[:, -1])
        cbar_ax.set_facecolor("#1a1a1a")
        cbar = fig.colorbar(last_im, cax=cbar_ax, ticks=RAIN_LEVELS, spacing="proportional")
        cbar.outline.set_edgecolor("#555555"); cbar.outline.set_linewidth(0.5)
        cbar_ax.yaxis.set_ticks_position("left"); cbar_ax.yaxis.set_label_position("left")
        cbar_ax.tick_params(axis="y", length=0, pad=4)
        plt.setp(cbar_ax.get_yticklabels(),
                 color="white", fontsize=7, fontfamily="monospace", fontweight="bold")
        cbar_ax_r = cbar_ax.twinx()
        cbar_ax_r.set_ylim(cbar_ax.get_ylim()); cbar_ax_r.set_yticks([])
        cbar_ax_r.set_ylabel("mm / h", color="#cccccc", fontsize=8, fontweight="bold",
                              labelpad=10, rotation=270, va="bottom")
        cbar_ax_r.spines[:].set_visible(False)

        for thr, badge, col in [
            (operational_thr, f"OPR\n{operational_thr:.0f}", "#ffee00"),
            (extreme_thr,     f"EXT\n{extreme_thr:.0f}",     "#ff6600"),
        ]:
            cbar_ax.axhline(y=thr, color=col, linewidth=1.2,
                            linestyle="--", alpha=0.9, xmin=-0.5, xmax=1.2, clip_on=False)
            cbar_ax.annotate(badge, xy=(1.15, thr),
                             xycoords=("axes fraction", "data"),
                             fontsize=6, fontweight="bold", color=col, va="center", ha="left",
                             bbox=dict(boxstyle="round,pad=0.25", fc="#1a1a1a", ec=col, lw=0.8))

    fig.legend(
        handles=[
            mpatches.Patch(facecolor=(1.0, 0.6, 0.0, 0.4),
                           label=f"Operational >={operational_thr:.0f} mm/h"),
            mpatches.Patch(facecolor=(1.0, 0.0, 1.0, 0.5),
                           label=f"Extreme >={extreme_thr:.0f} mm/h"),
        ],
        loc="lower center", ncol=2, fontsize=8, framealpha=0.3,
        facecolor="#333333", edgecolor="#555555", labelcolor="white",
        bbox_to_anchor=(0.48, 0.005)
    )

    os.makedirs(outdir, exist_ok=True)
    filepath = os.path.join(outdir, f'epoch_{epoch:03d}.png')
    fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    plt.close('all')
    del preds_cpu, targets_cpu, masks_cpu
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"  Visualization saved: {filepath}")
    mlflow.log_artifact(filepath, artifact_path=f'visualizations/{split_name}')
    return filepath

class HybridWeightedMAE(nn.Module):
    def __init__(self,
                 weight_threshold_1=3.0,  weight_value_1=3.0,
                 weight_threshold_2=7.0,  weight_value_2=7.0,
                 weight_threshold_3=15.0, weight_value_3=15.0,
                 weight_threshold_4=25.0, weight_value_4=25.0,
                 weight_threshold_5=35.0, weight_value_5=35.0):
        super().__init__()
        self.weight_threshold_1 = weight_threshold_1; self.weight_value_1 = weight_value_1
        self.weight_threshold_2 = weight_threshold_2; self.weight_value_2 = weight_value_2
        self.weight_threshold_3 = weight_threshold_3; self.weight_value_3 = weight_value_3
        self.weight_threshold_4 = weight_threshold_4; self.weight_value_4 = weight_value_4
        self.weight_threshold_5 = weight_threshold_5; self.weight_value_5 = weight_value_5

    def compute_weights(self, target_mmh):
        w = torch.ones_like(target_mmh)
        for thr, val in [
            (self.weight_threshold_1, self.weight_value_1),
            (self.weight_threshold_2, self.weight_value_2),
            (self.weight_threshold_3, self.weight_value_3),
            (self.weight_threshold_4, self.weight_value_4),
            (self.weight_threshold_5, self.weight_value_5),
        ]:
            w = torch.where(target_mmh >= thr,
                            torch.tensor(val, device=w.device, dtype=w.dtype), w)
        return w

    def forward(self, pred, target, mask):
        mask       = mask.bool()
        target_mmh = inverse_transform_torch(target)
        weights    = self.compute_weights(target_mmh)
        loss       = weights * torch.abs(pred - target)
        return loss[mask].mean() if mask.any() else loss.new_tensor(0.0)


class HybridWeightedMSE(nn.Module):
    """5-tier weighted MSE -- same threshold/weight structure as HybridWeightedMAE."""
    def __init__(self,
                 weight_threshold_1=3.0,  weight_value_1=3.0,
                 weight_threshold_2=7.0,  weight_value_2=7.0,
                 weight_threshold_3=15.0, weight_value_3=15.0,
                 weight_threshold_4=25.0, weight_value_4=25.0,
                 weight_threshold_5=35.0, weight_value_5=35.0):
        super().__init__()
        self.weight_threshold_1 = weight_threshold_1; self.weight_value_1 = weight_value_1
        self.weight_threshold_2 = weight_threshold_2; self.weight_value_2 = weight_value_2
        self.weight_threshold_3 = weight_threshold_3; self.weight_value_3 = weight_value_3
        self.weight_threshold_4 = weight_threshold_4; self.weight_value_4 = weight_value_4
        self.weight_threshold_5 = weight_threshold_5; self.weight_value_5 = weight_value_5

    def compute_weights(self, target_mmh):
        w = torch.ones_like(target_mmh)
        for thr, val in [
            (self.weight_threshold_1, self.weight_value_1),
            (self.weight_threshold_2, self.weight_value_2),
            (self.weight_threshold_3, self.weight_value_3),
            (self.weight_threshold_4, self.weight_value_4),
            (self.weight_threshold_5, self.weight_value_5),
        ]:
            w = torch.where(target_mmh >= thr,
                            torch.tensor(val, device=w.device, dtype=w.dtype), w)
        return w

    def forward(self, pred, target, mask):
        mask       = mask.bool()
        target_mmh = inverse_transform_torch(target)
        weights    = self.compute_weights(target_mmh)
        loss       = weights * (pred - target) ** 2
        return loss[mask].mean() if mask.any() else loss.new_tensor(0.0)


class GradientSharpnessLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, mask):
        # Convert to mm/h -- gradients are physically meaningful there
        pred_mm   = inverse_transform_torch(pred)
        target_mm = inverse_transform_torch(target)

        # Finite differences  (B, H, W-1) and (B, H-1, W)
        dx_pred = pred_mm[:, :, 1:] - pred_mm[:, :, :-1]
        dy_pred = pred_mm[:, 1:, :] - pred_mm[:, :-1, :]
        dx_gt   = target_mm[:, :, 1:] - target_mm[:, :, :-1]
        dy_gt   = target_mm[:, 1:, :] - target_mm[:, :-1, :]

        # Trim mask to match gradient dimensions
        mask_bool = mask.bool()
        mask_dx   = mask_bool[:, :, 1:]  & mask_bool[:, :, :-1]
        mask_dy   = mask_bool[:, 1:, :]  & mask_bool[:, :-1, :]

        if not mask_dx.any() and not mask_dy.any():
            return pred.new_tensor(0.0)

        loss_dx = torch.abs(dx_pred - dx_gt)[mask_dx].mean() \
                  if mask_dx.any() else pred.new_tensor(0.0)
        loss_dy = torch.abs(dy_pred - dy_gt)[mask_dy].mean() \
                  if mask_dy.any() else pred.new_tensor(0.0)

        return (loss_dx + loss_dy) * 0.5


# ============================================================
# VGG19 PERCEPTUAL LOSS
# ============================================================
class VGG19PerceptualLoss(nn.Module):
    _LAYER_MAP = {'relu1_2': 3, 'relu2_2': 8, 'relu3_4': 17, 'relu4_4': 26, 'relu5_4': 35}

    def __init__(self, layers=None, input_mode='repeat'):
        super().__init__()
        if layers is None:
            layers = ['relu2_2', 'relu3_4']
        for lname in layers:
            if lname not in self._LAYER_MAP:
                raise ValueError(f"Unknown perceptual layer '{lname}'")
        self.layers     = layers
        self.input_mode = input_mode

        import torchvision.models as tvm
        vgg     = tvm.vgg19(weights=tvm.VGG19_Weights.IMAGENET1K_V1)
        max_idx = max(self._LAYER_MAP[l] for l in layers) + 1
        self.vgg_features = nn.Sequential(*list(vgg.features.children())[:max_idx])
        for p in self.vgg_features.parameters():
            p.requires_grad = False

        self.register_buffer('vgg_mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('vgg_std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _prepare_input(self, x_log):
        mmh  = inverse_transform_torch(x_log)
        norm = (mmh / float(CLIP_MAX_MMH)).clamp(0.0, 1.0)
        rgb  = norm.unsqueeze(1).repeat(1, 3, 1, 1).float()
        return (rgb - self.vgg_mean) / self.vgg_std

    def _extract_features(self, x_rgb):
        features  = {}
        layer_set = {self._LAYER_MAP[l]: l for l in self.layers}
        out = x_rgb
        for idx, layer in enumerate(self.vgg_features):
            out = layer(out)
            if idx in layer_set:
                features[layer_set[idx]] = out
        return features

    def forward(self, pred, target, mask=None):
        if mask is not None and not mask.bool().any():
            return pred.new_tensor(0.0)
        pred_rgb, target_rgb = self._prepare_input(pred), self._prepare_input(target)
        with torch.no_grad():
            target_feats = self._extract_features(target_rgb)
        pred_feats = self._extract_features(pred_rgb)
        loss = pred.new_tensor(0.0)
        for lname in self.layers:
            loss = loss + torch.nn.functional.mse_loss(
                pred_feats[lname], target_feats[lname])
        return loss / len(self.layers)

def compute_multihorizon_loss(
        loss_fn, pred, target, mask, horizon_weights,
        perceptual_fn=None, perceptual_weight=0.0,
        gradient_fn=None, gradient_weight=0.0):
    
    n_horizons         = pred.shape[1]
    total_loss         = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    per_horizon_losses = []
    weights_sum        = sum(horizon_weights)

    for h_idx in range(n_horizons):
        pred_h   = pred[:, h_idx]
        target_h = target[:, h_idx]
        mask_h   = mask[:, h_idx]

        loss_h = loss_fn(pred_h, target_h, mask_h)

        if gradient_fn is not None and gradient_weight > 0.0:
            loss_h = loss_h + gradient_weight * gradient_fn(pred_h, target_h, mask_h)

        if perceptual_fn is not None and perceptual_weight > 0.0:
            loss_h = loss_h + perceptual_weight * perceptual_fn(pred_h, target_h, mask_h)

        total_loss = total_loss + horizon_weights[h_idx] * loss_h
        per_horizon_losses.append(loss_h.item())

    return total_loss / weights_sum, per_horizon_losses

def compute_psnr_mmh(pred, target, mask, max_val=CLIP_MAX_MMH):
    mask      = mask.bool()
    pred_mm   = inverse_transform_torch(pred)
    target_mm = inverse_transform_torch(target)
    if not mask.any():
        return 0.0
    mse = ((pred_mm[mask] - target_mm[mask]) ** 2).mean()
    if mse == 0:
        return float('inf')
    return (20 * torch.log10(
        torch.tensor(max_val, device=mse.device, dtype=mse.dtype) / torch.sqrt(mse)
    )).item()


def compute_ssim_mmh(pred, target, mask, max_val=CLIP_MAX_MMH):
    pred_mm  = inverse_transform_torch(pred).cpu().numpy()
    tgt_mm   = inverse_transform_torch(target).cpu().numpy()
    mask_np  = mask.bool().cpu().numpy()
    scores   = []
    for b in range(pred_mm.shape[0]):
        if mask_np[b].sum() < 100:
            continue
        p = np.nan_to_num(np.where(mask_np[b], pred_mm[b], np.nan), nan=0.0)
        t = np.nan_to_num(np.where(mask_np[b], tgt_mm[b],  np.nan), nan=0.0)
        scores.append(skimage_ssim(p, t, data_range=max_val, gaussian_weights=True,
                                   sigma=1.5, use_sample_covariance=False))
    return float(np.mean(scores)) if scores else float('nan')


def compute_metrics_per_horizon(pred, target, mask, horizons):
    metrics = {}
    for h_idx, h in enumerate(horizons):
        pred_h   = pred[:, h_idx]
        target_h = target[:, h_idx]
        mask_h   = mask[:, h_idx].bool()
        pred_mm  = inverse_transform_torch(pred_h)
        tgt_mm   = inverse_transform_torch(target_h)

        if mask_h.any():
            mae  = torch.abs(pred_mm[mask_h] - tgt_mm[mask_h]).mean()
            rmse = torch.sqrt(((pred_mm[mask_h] - tgt_mm[mask_h]) ** 2).mean())
        else:
            mae  = torch.abs(pred_mm - tgt_mm).mean()
            rmse = torch.sqrt(((pred_mm - tgt_mm) ** 2).mean())

        metrics[f't{h}'] = {
            'mae':  mae.item(),
            'rmse': rmse.item(),
            'ssim': compute_ssim_mmh(pred_h, target_h, mask_h),
            'psnr': compute_psnr_mmh(pred_h, target_h, mask_h),
        }
    return metrics

def build_scheduler(cfg, opt, train_loader):
    name = cfg.lr_scheduler.lower().strip()

    if name == 'none':
        print("  LR Scheduler : None\n")
        return None, 'none'

    elif name == 'one_cycle':
        steps_per_epoch = math.ceil(len(train_loader) / cfg.gradient_accumulation_steps)
        total_steps     = steps_per_epoch * cfg.num_epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=cfg.one_cycle_max_lr, total_steps=total_steps,
            pct_start=cfg.one_cycle_pct_start, anneal_strategy='cos',
            div_factor=cfg.one_cycle_max_lr / cfg.learning_rate, final_div_factor=1e4)
        return scheduler, 'one_cycle'

    elif name == 'cosine_annealing':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg.cosine_T_max, eta_min=cfg.cosine_eta_min)
        return scheduler, 'cosine_annealing'

    elif name == 'cosine_warm_restarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=cfg.cosine_T_0, T_mult=1, eta_min=cfg.cosine_eta_min)
        return scheduler, 'cosine_annealing'

    elif name == 'reduce_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode=cfg.plateau_mode, factor=cfg.plateau_factor,
            patience=cfg.plateau_patience, threshold=cfg.plateau_threshold)
        return scheduler, 'reduce_plateau'

    else:
        raise ValueError(f"Unknown lr_scheduler='{cfg.lr_scheduler}'")

def train_epoch(model, loader, loss_fn, opt, device, grad_clip,
                scaler, accumulation_steps, epoch,
                horizon_weights, horizons,
                scheduler=None, scheduler_type='none',
                gradient_fn=None, gradient_weight=0.0,
                perceptual_fn=None, perceptual_weight=0.0):
    model.train()
    total_loss           = 0.0
    horizon_losses_accum = {f't{h}': 0.0 for h in horizons}
    horizon_mae_accum    = {f't{h}': 0.0 for h in horizons}
    n             = 0
    total_batches = len(loader)
    last_h        = horizons[-1]

    opt.zero_grad()
    pbar = tqdm(enumerate(loader), total=total_batches,
                desc=f"  Train E{epoch:02d}", unit="batch",
                dynamic_ncols=True, leave=True)

    for batch_idx, (inputs, targets, masks) in pbar:
        inputs  = inputs.to(device)
        targets = targets.to(device)
        masks   = masks.to(device)

        with autocast('cuda', enabled=(scaler is not None)):
            pred = model(inputs)
            loss, per_horizon_losses = compute_multihorizon_loss(
                loss_fn, pred, targets, masks, horizon_weights,
                gradient_fn=gradient_fn, gradient_weight=gradient_weight,
                perceptual_fn=perceptual_fn, perceptual_weight=perceptual_weight)
            loss = loss / accumulation_steps

        if not torch.isfinite(loss):
            print(f"\n  [WARN] NaN/Inf loss at batch {batch_idx} -- skipping")
            opt.zero_grad()
            if scaler is not None:
                scaler.update()
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        is_accum = (batch_idx + 1) % accumulation_steps == 0
        is_last  = (batch_idx + 1) == total_batches

        if is_accum or is_last:
            if scaler is not None:
                scaler.unscale_(opt)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if torch.isfinite(grad_norm):
                    scaler.step(opt)
                else:
                    print(f"\n  [WARN] Non-finite grad norm at batch {batch_idx} -- skipping step")
                scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if torch.isfinite(grad_norm):
                    opt.step()
                else:
                    print(f"\n  [WARN] Non-finite grad norm at batch {batch_idx} -- skipping step")
            opt.zero_grad()
            if scheduler is not None and scheduler_type == 'one_cycle':
                scheduler.step()

        with torch.no_grad():
            metrics = compute_metrics_per_horizon(pred, targets, masks, horizons)

        total_loss += loss.item() * accumulation_steps
        for h_idx, h in enumerate(horizons):
            horizon_losses_accum[f't{h}'] += per_horizon_losses[h_idx]
            horizon_mae_accum[f't{h}']    += metrics[f't{h}']['mae']
        n += 1

        pbar.set_postfix({
            'loss':           f"{total_loss/n:.4f}",
            f't{last_h}_mae': f"{horizon_mae_accum[f't{last_h}']/n:.2f}mm/h",
            'lr':             f"{opt.param_groups[0]['lr']:.2e}",
        })

    return {
        'loss':           total_loss / n,
        'horizon_losses': {k: v / n for k, v in horizon_losses_accum.items()},
        'horizon_mae':    {k: v / n for k, v in horizon_mae_accum.items()},
    }

@torch.no_grad()
def validate(model, loader, loss_fn, device, horizon_weights, horizons, cfg,
             gradient_fn=None, gradient_weight=0.0,
             perceptual_fn=None, perceptual_weight=0.0):
    model.eval()
    total_loss            = 0.0
    horizon_losses_accum  = {f't{h}': 0.0 for h in horizons}
    horizon_metrics_accum = {
        f't{h}': {'mae': 0.0, 'rmse': 0.0, 'ssim': 0.0, 'ssim_count': 0,
                  'psnr': 0.0, 'psnr_count': 0}
        for h in horizons
    }
    contingency_op  = {f't{h}': {'H': 0, 'FA': 0, 'M': 0, 'CN': 0} for h in horizons}
    contingency_ext = {f't{h}': {'H': 0, 'FA': 0, 'M': 0, 'CN': 0} for h in horizons}
    total_pixels_per_horizon = {f't{h}': 0 for h in horizons}
    n      = 0
    last_h = horizons[-1]

    pbar = tqdm(enumerate(loader), total=len(loader), desc="  Val     ",
                unit="batch", dynamic_ncols=True, leave=True)

    for batch_idx, (inputs, targets, masks) in pbar:
        inputs  = inputs.to(device)
        targets = targets.to(device)
        masks   = masks.to(device)

        pred = model(inputs)
        loss, per_horizon_losses = compute_multihorizon_loss(
            loss_fn, pred, targets, masks, horizon_weights,
            gradient_fn=gradient_fn, gradient_weight=gradient_weight,
            perceptual_fn=perceptual_fn, perceptual_weight=perceptual_weight)
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

            pred_mm  = inverse_transform_torch(pred[:, h_idx]).cpu().numpy()
            gt_mm    = inverse_transform_torch(targets[:, h_idx]).cpu().numpy()
            mask_np  = masks[:, h_idx].cpu().numpy()

            for b in range(pred_mm.shape[0]):
                total_pixels_per_horizon[f't{h}'] += int((mask_np[b] > 0.5).sum())

                if cfg.compute_verification_metrics:
                    H, FA, M, CN = accumulate_contingency(
                        pred_mm[b], gt_mm[b], mask_np[b], cfg.ets_threshold_mmh)
                    contingency_op[f't{h}']['H']  += H
                    contingency_op[f't{h}']['FA'] += FA
                    contingency_op[f't{h}']['M']  += M
                    contingency_op[f't{h}']['CN'] += CN

                if cfg.compute_extreme_metrics:
                    H, FA, M, CN = accumulate_contingency(
                        pred_mm[b], gt_mm[b], mask_np[b], cfg.extreme_threshold_mmh)
                    contingency_ext[f't{h}']['H']  += H
                    contingency_ext[f't{h}']['FA'] += FA
                    contingency_ext[f't{h}']['M']  += M
                    contingency_ext[f't{h}']['CN'] += CN

            del pred_mm, gt_mm, mask_np

        n += 1
        ssim_val = metrics[f't{last_h}']['ssim']
        ssim_str = f"{ssim_val:.3f}" if not math.isnan(ssim_val) else "nan"
        pbar.set_postfix({
            'loss':            f"{loss.item():.4f}",
            f't{last_h}_mae':  f"{metrics[f't{last_h}']['mae']:.2f}mm/h",
            f't{last_h}_ssim': ssim_str,
        })

    avg_horizon_metrics = {}
    for h in horizons:
        acc        = horizon_metrics_accum[f't{h}']
        count      = acc['ssim_count']
        psnr_count = acc['psnr_count']
        avg_horizon_metrics[f't{h}'] = {
            'mae':  acc['mae']  / n,
            'rmse': acc['rmse'] / n,
            'ssim': acc['ssim'] / count      if count      > 0 else float('nan'),
            'psnr': acc['psnr'] / psnr_count if psnr_count > 0 else float('nan'),
        }

    ets_csi_metrics = {}
    mean_ets, mean_csi = 0.0, 0.0

    if cfg.compute_verification_metrics:
        ets_list, csi_list = [], []
        for h in horizons:
            c        = contingency_op[f't{h}']
            ets, csi = compute_ets_csi_from_totals(c['H'], c['FA'], c['M'], c['CN'])
            ets_csi_metrics[f't{h}'] = {'ets': ets, 'csi': csi}
            ets_list.append(ets); csi_list.append(csi)
        mean_ets = sum(ets_list) / len(ets_list)
        mean_csi = sum(csi_list) / len(csi_list)

    extreme_metrics = {}
    if cfg.compute_extreme_metrics:
        ext_ets_list = []
        for h in horizons:
            c                = contingency_ext[f't{h}']
            ets_ext, csi_ext = compute_ets_csi_from_totals(c['H'], c['FA'], c['M'], c['CN'])
            total_pix        = total_pixels_per_horizon[f't{h}']
            obs_ext          = c['H'] + c['M']
            base_rate        = obs_ext / total_pix if total_pix > 0 else 0.0
            extreme_metrics[f't{h}'] = {'ets': ets_ext, 'csi': csi_ext, 'base_rate': base_rate}
            ext_ets_list.append(ets_ext)

        if cfg.compute_verification_metrics and ext_ets_list:
            all_ets  = ets_list + ext_ets_list
            mean_ets = sum(all_ets) / len(all_ets)

    return {
        'loss':            total_loss / n,
        'horizon_losses':  {k: v / n for k, v in horizon_losses_accum.items()},
        'horizon_metrics': avg_horizon_metrics,
        'ets_csi_metrics': ets_csi_metrics,
        'extreme_metrics': extreme_metrics,
        'mean_ets':        mean_ets,
        'mean_csi':        mean_csi,
    }


def train(model, train_loader, val_loader, cfg, transform,
          operational_thr, extreme_thr, extreme_source, horizons, total_params):

    n_horizons = len(horizons)
    assert len(cfg.horizon_loss_weights) == n_horizons, (
        f"horizon_loss_weights has {len(cfg.horizon_loss_weights)} entries "
        f"but horizons has {n_horizons}."
    )

    print("\n" + "="*80)
    print("MULTI-HORIZON TRAINING -- SmaAt-UNet  [2H: t+15, t+30]")
    print("="*80)
    print(f"  Loss Type       : {cfg.loss_type.upper()}")
    print(f"  Loss Weights    : "
          f"{cfg.weight_threshold_1}mm/h->{cfg.weight_value_1}x  "
          f"{cfg.weight_threshold_2}mm/h->{cfg.weight_value_2}x  "
          f"{cfg.weight_threshold_3}mm/h->{cfg.weight_value_3}x  "
          f"{cfg.weight_threshold_4}mm/h->{cfg.weight_value_4}x  "
          f"{cfg.weight_threshold_5}mm/h->{cfg.weight_value_5}x")
    print(f"  Gradient Loss   : {'ENABLED weight=' + str(cfg.gradient_loss_weight) if cfg.use_gradient_loss else 'DISABLED'}")
    print(f"  Perceptual Loss : {'ENABLED weight=' + str(cfg.perceptual_weight) if cfg.use_perceptual_loss else 'DISABLED'}")
    print(f"  ETS@            : {cfg.ets_threshold_mmh:.0f} mm/h")
    print(f"  Extreme@        : {cfg.extreme_threshold_mmh:.0f} mm/h")
    print(f"  Horizons        : {horizons} min")
    print(f"  Horizon weights : {cfg.horizon_loss_weights}")
    print(f"  LR Scheduler    : {cfg.lr_scheduler}")
    print(f"  Mixed Precision : {cfg.use_mixed_precision}")
    print(f"  Resolution      : 256x256 @ 1km/pixel (PATCH)")
    print("="*80 + "\n")

    if cfg.loss_type == 'hybrid_weighted_mae':
        loss_fn = HybridWeightedMAE(
            cfg.weight_threshold_1, cfg.weight_value_1,
            cfg.weight_threshold_2, cfg.weight_value_2,
            cfg.weight_threshold_3, cfg.weight_value_3,
            cfg.weight_threshold_4, cfg.weight_value_4,
            cfg.weight_threshold_5, cfg.weight_value_5,
        ).to(cfg.device)
        print("HybridWeightedMAE [5 thresholds: 3/7/15/25/35mm/h -> 3/7/15/25/35x]\n")
    elif cfg.loss_type == 'hybrid_weighted_mse':
        loss_fn = HybridWeightedMSE(
            cfg.weight_threshold_1, cfg.weight_value_1,
            cfg.weight_threshold_2, cfg.weight_value_2,
            cfg.weight_threshold_3, cfg.weight_value_3,
            cfg.weight_threshold_4, cfg.weight_value_4,
            cfg.weight_threshold_5, cfg.weight_value_5,
        ).to(cfg.device)
        print("HybridWeightedMSE [5 thresholds: 3/7/15/25/35mm/h -> 3/7/15/25/35x]\n")
    else:
        raise ValueError(f"Unknown loss_type: {cfg.loss_type}")

    gradient_fn     = None
    gradient_weight = 0.0
    if cfg.use_gradient_loss:
        gradient_fn     = GradientSharpnessLoss().to(cfg.device)
        gradient_weight = cfg.gradient_loss_weight
        print(f"Gradient Sharpness Loss ENABLED  weight={gradient_weight}\n")
    else:
        print("Gradient Sharpness Loss DISABLED\n")

    perceptual_fn     = None
    perceptual_weight = 0.0
    if cfg.use_perceptual_loss:
        perceptual_fn = VGG19PerceptualLoss(
            layers=cfg.perceptual_layers, input_mode=cfg.perceptual_input_mode
        ).to(cfg.device)
        perceptual_weight = cfg.perceptual_weight
        print(f"VGG19 Perceptual Loss ENABLED  layers={cfg.perceptual_layers}  "
              f"weight={cfg.perceptual_weight}\n")
    else:
        print("VGG19 Perceptual Loss DISABLED\n")

    opt = optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                      betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)
    print(f"AdamW | LR={cfg.learning_rate:.2e}")

    scheduler, scheduler_type = build_scheduler(cfg, opt, train_loader)
    scaler = GradScaler() if cfg.use_mixed_precision else None
    if scaler:
        print("Mixed Precision ENABLED\n")

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.train_viz_dir,  exist_ok=True)
    os.makedirs(cfg.val_viz_dir,    exist_ok=True)

    start_epoch                = 1
    best_val_loss              = float('inf')
    best_mean_ets              = -float('inf')
    epochs_without_improvement = 0
    loss_history               = []
    ets_history                = []

    if cfg.resume_from and os.path.exists(cfg.resume_from):
        print(f"\nRESUMING FROM: {cfg.resume_from}")
        ckpt = torch.load(cfg.resume_from, map_location=cfg.device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch   = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        best_mean_ets = ckpt.get('best_mean_ets', -float('inf'))
        loss_history  = ckpt.get('loss_history', [])
        ets_history   = ckpt.get('ets_history', [])
        print(f"Resumed from epoch {ckpt['epoch']} | "
              f"Best Val Loss: {best_val_loss:.6f} | Best ETS: {best_mean_ets:.4f}\n")

    mlflow_log_config(cfg, horizons,
                      n_train=len(train_loader.dataset),
                      n_val=len(val_loader.dataset),
                      total_params=total_params)

    last_h = horizons[-1]

    for epoch in range(start_epoch, cfg.num_epochs + 1):
        t0 = time.time()
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch}/{cfg.num_epochs}")
        print(f"{'='*80}")

        current_weights = get_horizon_weights(epoch, cfg.horizon_loss_weights)
        print(f"  Horizon weights: {current_weights}", flush=True)

        train_metrics = train_epoch(
            model, train_loader, loss_fn, opt, cfg.device,
            cfg.grad_clip, scaler, cfg.gradient_accumulation_steps,
            epoch, current_weights, horizons,
            scheduler=scheduler, scheduler_type=scheduler_type,
            gradient_fn=gradient_fn, gradient_weight=gradient_weight,
            perceptual_fn=perceptual_fn, perceptual_weight=perceptual_weight,
        )

        val_metrics = validate(
            model, val_loader, loss_fn, cfg.device,
            current_weights, horizons, cfg,
            gradient_fn=gradient_fn, gradient_weight=gradient_weight,
            perceptual_fn=perceptual_fn, perceptual_weight=perceptual_weight,
        )

        dt       = time.time() - t0
        val_loss = val_metrics['loss']
        mean_ets = val_metrics['mean_ets']
        mean_csi = val_metrics['mean_csi']

        if scheduler is not None:
            if scheduler_type == 'cosine_annealing':
                scheduler.step()
            elif scheduler_type == 'reduce_plateau':
                scheduler.step(val_loss)

        current_lr = opt.param_groups[0]['lr']

        print(f"\n  Results (Time: {dt:.1f}s | LR: {current_lr:.6e}):")
        print(f"    Train Loss : {train_metrics['loss']:.6f}")
        print(f"    Val   Loss : {val_loss:.6f}")
        print(f"    Avg ETS    : {mean_ets:.4f}  "
              f"(@{cfg.ets_threshold_mmh:.0f}+@{cfg.extreme_threshold_mmh:.0f}mm/h avg)")
        print(f"    Avg CSI@{cfg.ets_threshold_mmh:.0f}mm/h : {mean_csi:.4f}")

        loss_history.append(val_loss)
        ets_history.append(mean_ets)
        if len(loss_history) > 1:
            print(f"    Loss Change: {loss_history[-1] - loss_history[-2]:+.6f}")

        is_best = val_loss < best_val_loss - cfg.min_delta

        mlflow_log_epoch(epoch, train_metrics, val_metrics, current_lr, dt, horizons,
                         is_best, cfg.ets_threshold_mmh, cfg.extreme_threshold_mmh)

        if should_report_ets(epoch, cfg.ets_report_epochs) and cfg.compute_verification_metrics:
            print(f"\n  Operational Verification @ {cfg.ets_threshold_mmh:.0f} mm/h:")
            print(f"  {'Horizon':<10} {'MAE(mm/h)':<12} {'RMSE(mm/h)':<12} {'ETS':<10} {'CSI':<10}")
            print(f"  {'-'*54}")
            ets_vals, csi_vals = [], []
            for h in horizons:
                key    = f't{h}'
                h_met  = val_metrics['horizon_metrics'][key]
                h_ets  = val_metrics['ets_csi_metrics'][key]
                marker = " <- PRIMARY" if h == last_h else ""
                print(f"  t+{h:<7} {h_met['mae']:<12.2f} {h_met['rmse']:<12.2f} "
                      f"{h_ets['ets']:<10.4f} {h_ets['csi']:<10.4f}{marker}")
                ets_vals.append(h_ets['ets']); csi_vals.append(h_ets['csi'])
            print(f"  {'AVG':<10} {'---':<12} {'---':<12} "
                  f"{sum(ets_vals)/len(ets_vals):<10.4f} "
                  f"{sum(csi_vals)/len(csi_vals):<10.4f}  <- AVG")

            if cfg.compute_extreme_metrics:
                print(f"\n  Extreme Core @ {cfg.extreme_threshold_mmh:.0f} mm/h (secondary):")
                print(f"  {'Horizon':<10} {'ETS':<10} {'CSI':<10} {'BaseRate':<12}")
                print(f"  {'-'*42}")
                ext_ets_vals = []
                for h in horizons:
                    key   = f't{h}'
                    h_ext = val_metrics['extreme_metrics'][key]
                    ext_ets_vals.append(h_ext['ets'])
                    print(f"  t+{h:<7} {h_ext['ets']:<10.4f} "
                          f"{h_ext['csi']:<10.4f} {h_ext['base_rate']:<12.6f}")
                print(f"  {'AVG':<10} {sum(ext_ets_vals)/len(ext_ets_vals):<10.4f} "
                      f"{'---':<10} {'---':<12}  <- AVG")

        last_ckpt_path = os.path.join(cfg.checkpoint_dir, 'last.pth')
        torch.save({
            'epoch':                 epoch,
            'model_state_dict':      model.state_dict(),
            'optimizer_state_dict':  opt.state_dict(),
            'train_loss':            train_metrics['loss'],
            'val_loss':              val_loss,
            'best_val_loss':         best_val_loss,
            'mean_ets':              mean_ets,
            'mean_csi':              mean_csi,
            'val_ets_csi_metrics':   val_metrics['ets_csi_metrics'],
            'val_extreme_metrics':   val_metrics['extreme_metrics'],
            'val_horizon_metrics':   val_metrics['horizon_metrics'],
            'loss_history':          loss_history,
            'ets_history':           ets_history,
            'horizons':              horizons,
            'operational_threshold': operational_thr,
            'extreme_threshold':     extreme_thr,
            'softlog_eps':           SOFTLOG_EPS,
            'softlog_log_norm':      SOFTLOG_LOG_NORM,
            'softlog_zerovalue':     SOFTLOG_ZEROVALUE,
        }, last_ckpt_path)
        mlflow.log_artifact(last_ckpt_path, artifact_path='checkpoints')

        if epoch % cfg.viz_every_n_epochs == 0 or epoch == start_epoch:
            print(f"\n  Creating visualizations (epoch {epoch})...")
            save_visualizations(
                model, val_loader, epoch, transform, cfg,
                operational_thr, extreme_thr, horizons,
                cfg.val_viz_dir, "val", cfg.device)

        if is_best:
            best_val_loss              = val_loss
            epochs_without_improvement = 0

            best_ckpt_path = os.path.join(cfg.checkpoint_dir, 'best.pth')
            torch.save({
                'epoch':               epoch,
                'model_state_dict':    model.state_dict(),
                'val_loss':            val_loss,
                'best_val_loss':       val_loss,
                'mean_ets':            mean_ets,
                'mean_csi':            mean_csi,
                'horizons':            horizons,
                'val_ets_csi_metrics': val_metrics['ets_csi_metrics'],
                'val_extreme_metrics': val_metrics['extreme_metrics'],
                'softlog_eps':         SOFTLOG_EPS,
                'softlog_log_norm':    SOFTLOG_LOG_NORM,
                'softlog_zerovalue':   SOFTLOG_ZEROVALUE,
            }, best_ckpt_path)
            mlflow.log_artifact(best_ckpt_path, artifact_path='checkpoints')

            ets_tag = f'op{int(cfg.ets_threshold_mmh)}mmh'
            mlflow.log_metrics({
                'best/val_loss':       val_loss,
                f'best/ets_{ets_tag}': mean_ets,
                f'best/csi_{ets_tag}': mean_csi,
                'best/epoch':          float(epoch),
            }, step=epoch)
            print(f"\n  *** NEW BEST MODEL!  "
                  f"Val Loss={val_loss:.4f} | ETS={mean_ets:.4f} | CSI={mean_csi:.4f} ***")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epoch(s) "
                  f"(best: {best_val_loss:.4f})")
            if epochs_without_improvement >= cfg.patience:
                print("\nEARLY STOPPING")
                break

        if cfg.compute_verification_metrics and mean_ets > best_mean_ets:
            best_mean_ets = mean_ets
            best_ets_ckpt_path = os.path.join(cfg.checkpoint_dir, 'best_ets.pth')
            torch.save({
                'epoch':               epoch,
                'model_state_dict':    model.state_dict(),
                'val_loss':            val_loss,
                'best_val_loss':       best_val_loss,
                'mean_ets':            mean_ets,
                'best_mean_ets':       best_mean_ets,
                'mean_csi':            mean_csi,
                'horizons':            horizons,
                'val_ets_csi_metrics': val_metrics['ets_csi_metrics'],
                'val_extreme_metrics': val_metrics['extreme_metrics'],
                'softlog_eps':         SOFTLOG_EPS,
                'softlog_log_norm':    SOFTLOG_LOG_NORM,
                'softlog_zerovalue':   SOFTLOG_ZEROVALUE,
            }, best_ets_ckpt_path)
            mlflow.log_artifact(best_ets_ckpt_path, artifact_path='checkpoints')

            ets_tag = f'op{int(cfg.ets_threshold_mmh)}mmh'
            mlflow.log_metrics({
                'best_ets/mean_ets':       mean_ets,
                'best_ets/val_loss':       val_loss,
                f'best_ets/csi_{ets_tag}': mean_csi,
                'best_ets/epoch':          float(epoch),
            }, step=epoch)
            print(f"  *** NEW BEST ETS MODEL!  "
                  f"ETS={mean_ets:.4f} | Val Loss={val_loss:.4f} -> best_ets.pth ***")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE -- Best Val Loss: {best_val_loss:.4f} | "
          f"Best ETS: {best_mean_ets:.4f}")
    print(f"{'='*80}\n")

    ets_tag = f'op{int(cfg.ets_threshold_mmh)}mmh'
    mlflow.log_metrics({
        'final/best_val_loss':       best_val_loss,
        'final/best_mean_ets':       best_mean_ets,
        f'final/best_ets_{ets_tag}': max(ets_history) if ets_history else 0.0,
        'final/total_epochs':        float(epoch),
    })


if __name__ == "__main__":
    cfg = MultiHorizonTrainingConfig()

    print("\n" + "="*80)
    print("MULTI-HORIZON TRAINING -- SmaAt-UNet  [2H: t+15, t+30]")
    print("="*80)
    print(f"  Device        : {cfg.device}")
    print(f"  Loss Function : {cfg.loss_type}")
    print(f"  Gradient Loss : {cfg.use_gradient_loss} (weight={cfg.gradient_loss_weight})")
    print(f"  Resolution    : 256x256 @ 1km/pixel (native patch)")
    print(f"  Train CSV     : {cfg.metadata_train}")
    print(f"  Val CSV       : {cfg.metadata_val}")
    print(f"  LR Scheduler  : {cfg.lr_scheduler}")

    config     = load_config(cfg.config_yml)
    horizons   = compute_forecast_horizons(config.temporal.radar_lead_minutes)
    n_horizons = len(horizons)

    print(f"  Horizons      : {horizons}")
    assert len(cfg.horizon_loss_weights) == n_horizons, (
        f"horizon_loss_weights has {len(cfg.horizon_loss_weights)} entries "
        f"but config gives {n_horizons} horizons ({horizons}). "
        f"Update horizon_loss_weights in MultiHorizonTrainingConfig."
    )

    operational_thr, extreme_thr, extreme_source = load_two_thresholds(cfg)
    verify_clip_config(CLIP_MAX_MMH)

    policy = DataPolicy(config.data_policy.nan_handling, mask_nans=config.data_policy.mask_nans)

    print("\nLoading datasets...")
    train_dataset = SatelliteRadarPatchDataset(
        config=config, metadata_csv=cfg.metadata_train,
        mask_nans=policy.mask_nans, validate_files=False, use_cache=False
    )
    val_dataset = SatelliteRadarPatchDataset(
        config=config, metadata_csv=cfg.metadata_val,
        mask_nans=policy.mask_nans, validate_files=False, use_cache=False
    )
    print(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,}")

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=False,
        persistent_workers=True, prefetch_factor=cfg.prefetch_factor
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=cfg.prefetch_factor
    )

    sample_input, _, _  = train_dataset[0]
    n_channels          = sample_input.shape[0]
    n_timesteps         = config.temporal.history_minutes // config.satellite.cadence_minutes
    n_channels_per_step = len(config.satellite.channels) + 1   # CH7 + CH9 + radar = 3
    expected_channels   = n_timesteps * n_channels_per_step

    print(f"\nBuilding Model... [SmaAt-UNet]")
    print(f"   Input shape:        {sample_input.shape}")
    print(f"   n_channels (flat):  {n_channels} (expected {expected_channels})")
    print(f"   n_horizons:         {n_horizons}")
    print(f"   kernels_per_layer:  {cfg.smaat_kernels_per_layer}")
    print(f"   bilinear:           {cfg.smaat_bilinear}")
    print(f"   reduction_ratio:    {cfg.smaat_reduction_ratio}")
    assert n_channels == expected_channels, \
        f"Channel mismatch: got {n_channels}, expected {expected_channels}"

    model = SmaAt_UNet(
        n_channels        = n_channels,
        n_classes         = n_horizons,
        kernels_per_layer = cfg.smaat_kernels_per_layer,
        bilinear          = cfg.smaat_bilinear,
        reduction_ratio   = cfg.smaat_reduction_ratio,
    )
    model = model.to(cfg.device)

    if torch.cuda.device_count() > 1:
        print(f"   Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,}")

    transform = SoftLogTransform(inverse=True)

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment)

    with mlflow.start_run(run_name=cfg.mlflow_run_name):
        print(f"\nMLflow run started: {mlflow.active_run().info.run_id}")
        print(f"   Experiment : {cfg.mlflow_experiment}")
        print(f"   Tracking   : {cfg.mlflow_tracking_uri}\n")

        train(model, train_loader, val_loader, cfg, transform,
              operational_thr, extreme_thr, extreme_source, horizons, total_params)

    print("\nALL DONE")
    print(f"View results: mlflow ui --backend-store-uri {cfg.mlflow_tracking_uri}")