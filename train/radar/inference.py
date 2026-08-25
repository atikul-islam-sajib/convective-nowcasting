import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F

from skimage.morphology import closing, remove_small_objects, disk
from skimage.metrics import structural_similarity
from scipy.ndimage import label
from einops import rearrange
from omegaconf import OmegaConf

from src.models.smaat_unet.SmaAt_UNet import SmaAt_UNet
from src.models.convlstm.net_params_multihorizon import (
    convlstm_encoder_params, get_convlstm_decoder_params)
from src.models.convlstm.model import ED
from src.models.convlstm.encoder import Encoder
from src.models.convlstm.decoder import Decoder
from src.models.simvp.model import SimVP
from src.models.simvp.modules import ConvSC, Inception
from src.models.earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformerModel
from src.models.unet.unet_model import UNet
from src.models.vptr.model.VPTR_modules import VPTREnc, VPTRDec, VPTRFormerNAR


CLIP_MAX_MMH     = 128.0
SOFTLOG_EPS      = 1.0
SOFTLOG_LOG_NORM = np.log10(CLIP_MAX_MMH + SOFTLOG_EPS)

RAIN_LEVELS = [0.1, 1, 2, 5, 10, 15, 20, 30, 40, 60, 100]
RAIN_COLORS = [
    "#1a5c1a", "#22aa22", "#55cc22", "#ffee00", "#ffaa00",
    "#ff6600", "#ff2200", "#cc0000", "#aa0077", "#ff00ff",
]
RAIN_CMAP = ListedColormap(RAIN_COLORS)
RAIN_NORM = BoundaryNorm(RAIN_LEVELS, RAIN_CMAP.N)
BG_COLOR  = "#888888"


MODELS = [
    {'key': 'smaat_unet',  'label': 'SmaAt-UNet',  'color': '#4488ff'},
    {'key': 'convlstm',    'label': 'ConvLSTM',     'color': '#ff8844'},
    {'key': 'simvp',       'label': 'SimVP',         'color': '#44cc88'},
    {'key': 'earthformer', 'label': 'EarthFormer',  'color': '#cc44ff'},
    {'key': 'vptr',        'label': 'VPTR',          'color': '#ff4488'},
    {'key': 'ensemble',    'label': 'Ensemble',      'color': '#ffffff'},
]


REAL_MODEL_KEYS = [m['key'] for m in MODELS if m['key'] != 'ensemble']


class InferenceConfig:

    out_dir      = 'INFERENCE_FULLIMAGE_COMPARISON_RADAR'
    metadata_csv = 'metadata_patch/test_fullimage_summer.csv'

    detailed_metrics_csv    = os.path.join(out_dir, 'dl_fullimage_metrics.csv')
    mean_metrics_csv        = os.path.join(out_dir, 'dl_fullimage_mean_metrics.csv')
    pooled_mean_metrics_csv = os.path.join(out_dir, 'dl_fullimage_pooled_mean_metrics.csv')
    simple_mean_metrics_csv = os.path.join(out_dir, 'dl_fullimage_simple_mean_metrics.csv')

    radar_base = '/home/fe/sajib/scratch/weather-data/radar_de'

    num_in_frames  = 5
    stride_minutes = 5
    horizons       = [15, 30, 45, 60]

    smaat_checkpoint       = 'CHECKPOINTS_SMAATUNET_PATCH_1KM_SUMMER_2H_2015_to_2024_t15_to_t60_RADAR/best_ets.pth'
    convlstm_checkpoint    = 'CHECKPOINTS_CONVLSTM_PATCH_1KM_SUMMER_2H_2015_to_2024_t15_to_t60_RADAR/best_ets.pth'
    simvp_checkpoint       = 'CHECKPOINTS_SIMVP_PATCH_1KM_SUMMER_2H_2015_to_2024_t15_to_t60_RADAR/best_ets.pth'
    earthformer_checkpoint = 'CHECKPOINTS_EARTHFORMER_PATCH_1KM_SUMMER_2H_2015_to_2024_t15_to_t60_RADAR/best_ets.pth'
    vptr_checkpoint        = 'CHECKPOINTS_VPTR_PATCH_1KM_SUMMER_2H_2015_to_2024_t15_to_t60_RADAR/best_ets.pth'
    earthformer_config_yml = 'config/earthformer_nowcast.yaml'

    smaat_kernels_per_layer = 2
    smaat_bilinear          = True
    smaat_reduction_ratio   = 16

    hid_S        = 64
    hid_T        = 256
    N_S          = 4
    N_T          = 8
    incep_ker    = [3, 5, 7, 11]
    simvp_groups = 8

    unet_bilinear = True

    vptr_feat_dim           = 192
    vptr_n_downsampling     = 3
    vptr_encH               = 32
    vptr_encW               = 32
    vptr_d_model            = 192
    vptr_nhead              = 4
    vptr_num_encoder_layers = 2
    vptr_num_decoder_layers = 2
    vptr_dropout            = 0.1
    vptr_window_size        = 4
    vptr_spatial_ffn_ratio  = 4
    vptr_rpe                = True

    patch_size    = 256
    patch_overlap = 64
    patch_blend   = 'cosine'

    num_samples          = 5

    operational_thr      = 15.0
    extreme_thr          = 35.0
    storm_min_pixels     = 200
    morphology_disk_size = 4
    storm_min_area_km2   = 200.0
    storm_max_area_km2   = 10000.0
    pixel_area_km2       = 1.0

    show_storm_metrics = False

    rain_threshold      = 0.1
    zerovalue_db        = -15.0
    psnr_data_range      = CLIP_MAX_MMH

    csi_threshold_mmh      = [5.0, 15.0]
    categorical_thresholds = [5.0, 15.0]

    ensemble_weights = {"smaat_unet": 1.0, "convlstm": 1.0, "simvp": 1.0, "earthformer": 1.0, "unet": 1.0, "vptr": 1.0}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


def radar_path(base, dt):
    return os.path.join(base, dt.strftime('%y%m'), dt.strftime('%d'),
                        dt.strftime('%y%m%d_%H%M') + '.npy')


def get_history_times(current_time, num_frames, stride_minutes):
    start = current_time - (num_frames - 1) * timedelta(minutes=stride_minutes)
    return [start + timedelta(minutes=stride_minutes) * i for i in range(num_frames)]


def get_target_times(current_time, horizons_minutes):
    return [current_time + timedelta(minutes=h) for h in horizons_minutes]


def softlog_transform(x_mmh):
    return np.log10(np.clip(x_mmh, 0.0, CLIP_MAX_MMH) + SOFTLOG_EPS) / SOFTLOG_LOG_NORM


def inverse_transform_np(y_log):
    return np.clip(10.0 ** (y_log * SOFTLOG_LOG_NORM) - SOFTLOG_EPS, 0.0, CLIP_MAX_MMH)


def rainrate_to_db(rain, threshold=0.1, zerovalue=-15.0):
    rain = np.asarray(rain, dtype=np.float32)
    result = np.full_like(rain, zerovalue, dtype=np.float32)
    wet = rain >= threshold
    result[wet] = 10.0 * np.log10(rain[wet])
    return result


def contingency_counts(pred, target, threshold):
    pred_event = pred >= threshold
    target_event = target >= threshold
    hits = np.sum(pred_event & target_event)
    misses = np.sum((~pred_event) & target_event)
    false_alarms = np.sum(pred_event & (~target_event))
    correct_negatives = np.sum((~pred_event) & (~target_event))
    return float(hits), float(misses), float(false_alarms), float(correct_negatives)


def csi_from_counts(h, m, fa):
    denom = h + m + fa
    return float('nan') if denom == 0 else h / denom


def ets_from_counts(h, m, fa, cn):
    total = h + m + fa + cn
    if total == 0:
        return float('nan')
    random_hits = (h + m) * (h + fa) / total
    denom = h + m + fa - random_hits
    return float('nan') if denom == 0 else (h - random_hits) / denom


def compute_psnr(pred, target, data_range):
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float('nan')
    return 10.0 * np.log10(data_range ** 2 / mse)


def compute_metrics(prediction, target, mask, cfg):
    valid = mask.astype(bool) & np.isfinite(prediction) & np.isfinite(target)
    if not np.any(valid):
        return None
    p = prediction[valid]
    y = target[valid]
    mae = np.mean(np.abs(p - y))
    mse = np.mean((p - y) ** 2)
    rmse = np.sqrt(mse)
    psnr = compute_psnr(p, y, cfg.psnr_data_range)
    if valid.sum() < 100:
        ssim = float('nan')
    else:
        pred_ssim = np.where(valid, prediction, 0.0)
        target_ssim = np.where(valid, target, 0.0)
        _, ssim_map = structural_similarity(
            target_ssim, pred_ssim, data_range=cfg.psnr_data_range,
            gaussian_weights=True, sigma=1.5,
            use_sample_covariance=False, full=True)
        ssim = float(ssim_map[valid].mean())

    result = {'MAE': float(mae), 'MSE': float(mse), 'RMSE': float(rmse),
              'PSNR': float(psnr), 'SSIM': float(ssim)}

    csi_thresholds = cfg.csi_threshold_mmh
    if np.isscalar(csi_thresholds):
        csi_thresholds = [csi_thresholds]

    all_needed = set(csi_thresholds) | set(cfg.categorical_thresholds)
    counts_by_threshold = {threshold: contingency_counts(p, y, threshold) for threshold in all_needed}

    for threshold in csi_thresholds:
        suffix = str(int(threshold)) if float(threshold).is_integer() else str(threshold)
        h, m, fa, cn = counts_by_threshold[threshold]
        result[f'CSI@{suffix}'] = csi_from_counts(h, m, fa)
        result[f'THR{suffix}_csi_hits'] = h
        result[f'THR{suffix}_csi_misses'] = m
        result[f'THR{suffix}_csi_fa'] = fa
        result[f'THR{suffix}_csi_cn'] = cn

    for threshold in cfg.categorical_thresholds:
        suffix = str(int(threshold)) if float(threshold).is_integer() else str(threshold)
        h, m, fa, cn = counts_by_threshold[threshold]
        result[f'ETS@{suffix}'] = ets_from_counts(h, m, fa, cn)
        result[f'THR{suffix}_ets_hits'] = h
        result[f'THR{suffix}_ets_misses'] = m
        result[f'THR{suffix}_ets_fa'] = fa
        result[f'THR{suffix}_ets_cn'] = cn

    return result


def compute_pooled_mean(metrics_df, cfg):
    ratio_cols = ['MAE', 'MSE', 'RMSE', 'PSNR', 'SSIM']
    csi_thresholds = cfg.csi_threshold_mmh
    if np.isscalar(csi_thresholds):
        csi_thresholds = [csi_thresholds]

    rows = []
    for (model, horizon), g in metrics_df.groupby(['model', 'horizon_min']):
        row = {'model': model, 'horizon_min': horizon, 'n_samples': len(g)}
        for col in ratio_cols:
            row[col] = g[col].mean()

        csi_vals = []
        for threshold in csi_thresholds:
            suffix = str(int(threshold)) if float(threshold).is_integer() else str(threshold)
            h = g[f'THR{suffix}_csi_hits'].sum()
            m = g[f'THR{suffix}_csi_misses'].sum()
            fa = g[f'THR{suffix}_csi_fa'].sum()
            csi = csi_from_counts(h, m, fa)
            row[f'CSI@{suffix}'] = csi
            csi_vals.append(csi)

        ets_vals = []
        for threshold in cfg.categorical_thresholds:
            suffix = str(int(threshold)) if float(threshold).is_integer() else str(threshold)
            h = g[f'THR{suffix}_ets_hits'].sum()
            m = g[f'THR{suffix}_ets_misses'].sum()
            fa = g[f'THR{suffix}_ets_fa'].sum()
            cn = g[f'THR{suffix}_ets_cn'].sum()
            ets = ets_from_counts(h, m, fa, cn)
            row[f'ETS@{suffix}'] = ets
            ets_vals.append(ets)

        row['CSI-M'] = float(np.nanmean(csi_vals)) if csi_vals else float('nan')
        row['ETS-M'] = float(np.nanmean(ets_vals)) if ets_vals else float('nan')
        rows.append(row)
    return pd.DataFrame(rows)


def compute_simple_mean(metrics_df, cfg):
    csi_thresholds = cfg.csi_threshold_mmh
    if np.isscalar(csi_thresholds):
        csi_thresholds = [csi_thresholds]

    value_cols = ['MAE', 'MSE', 'RMSE', 'PSNR', 'SSIM']
    value_cols += [
        f'CSI@{str(int(t)) if float(t).is_integer() else str(t)}'
        for t in csi_thresholds
    ]
    value_cols += [
        f'ETS@{str(int(t)) if float(t).is_integer() else str(t)}'
        for t in cfg.categorical_thresholds
    ]
    value_cols = [c for c in value_cols if c in metrics_df.columns]

    rows = []
    for (model, horizon), g in metrics_df.groupby(['model', 'horizon_min']):
        row = {'model': model, 'horizon_min': horizon, 'n_samples': len(g)}
        for col in value_cols:
            row[col] = float(np.nanmean(g[col]))

        csi_cols = [c for c in value_cols if c.startswith('CSI@')]
        ets_cols = [c for c in value_cols if c.startswith('ETS@')]
        row['CSI-M'] = float(np.nanmean(g[csi_cols].values)) if csi_cols else float('nan')
        row['ETS-M'] = float(np.nanmean(g[ets_cols].values)) if ets_cols else float('nan')
        rows.append(row)
    return pd.DataFrame(rows)


def _cosine_window_1d(size, device):
    t = torch.linspace(0.0, 1.0, size, device=device)
    return 0.5 - 0.5 * torch.cos(2.0 * np.pi * t)


def _make_blend_window(patch_size, device):
    w1d = _cosine_window_1d(patch_size, device)
    return w1d.unsqueeze(0) * w1d.unsqueeze(1)


def _get_patch_starts(full_size, patch_size, overlap):
    stride = patch_size - overlap
    starts = list(range(0, full_size - patch_size, stride))
    last   = full_size - patch_size
    if not starts or starts[-1] != last:
        starts.append(last)
    starts = [max(0, s) for s in starts]
    return sorted(set(starts))


def patch_inference(model, input_tensor, n_horizons, patch_size, overlap, blend, device):
    input_tensor = input_tensor.to(device)
    _, C, H, W   = input_tensor.shape

    if H == patch_size and W == patch_size:
        with torch.no_grad():
            return model(input_tensor).cpu()

    pad_h = max(0, patch_size - H)
    pad_w = max(0, patch_size - W)
    if pad_h > 0 or pad_w > 0:
        input_tensor = F.pad(input_tensor, (0, pad_w, 0, pad_h), mode='reflect')
        _, _, H, W   = input_tensor.shape

    accum  = torch.zeros(1, n_horizons, H, W, device=device)
    weight = torch.zeros(1, 1,          H, W, device=device)

    blend_win = _make_blend_window(patch_size, device).unsqueeze(0).unsqueeze(0)

    row_starts = _get_patch_starts(H, patch_size, overlap)
    col_starts = _get_patch_starts(W, patch_size, overlap)
    n_patches  = len(row_starts) * len(col_starts)

    for r in row_starts:
        for c in col_starts:
            patch = input_tensor[:, :, r:r+patch_size, c:c+patch_size]
            with torch.no_grad():
                pred_patch = model(patch)
            if blend == 'cosine':
                w = blend_win.expand(1, n_horizons, patch_size, patch_size)
                accum[:, :, r:r+patch_size, c:c+patch_size]  += pred_patch * w
                weight[:, :, r:r+patch_size, c:c+patch_size] += blend_win
            else:
                accum[:, :, r:r+patch_size, c:c+patch_size]  += pred_patch
                weight[:, :, r:r+patch_size, c:c+patch_size] += 1.0

    stitched = (accum / weight.clamp(min=1e-6)).cpu()
    if pad_h > 0 or pad_w > 0:
        stitched = stitched[:, :, :H - pad_h, :W - pad_w]
    return stitched


def run_model(model, input_tensor, n_horizons, cfg):
    if cfg.patch_size is not None:
        return patch_inference(model, input_tensor, n_horizons,
                               cfg.patch_size, cfg.patch_overlap,
                               cfg.patch_blend, cfg.device)
    with torch.no_grad():
        return model(input_tensor.to(cfg.device)).cpu()


class ConvLSTMWrapper(nn.Module):
    def __init__(self, num_channels, num_timesteps, n_horizons):
        super().__init__()
        self.num_channels  = num_channels
        self.num_timesteps = num_timesteps
        encoder        = Encoder(convlstm_encoder_params[0], convlstm_encoder_params[1])
        decoder_params = get_convlstm_decoder_params(n_horizons)
        decoder        = Decoder(decoder_params[0], decoder_params[1])
        self.model     = ED(encoder, decoder)

    def forward(self, x):
        B, TC, H, W = x.shape
        x   = x.view(B, self.num_timesteps, self.num_channels, H, W)
        out = self.model(x)
        return out[:, 0, :, :, :].contiguous()


class SimVPWrapper(nn.Module):
    def __init__(self, num_channels, num_timesteps, n_horizons,
                 hid_S=64, hid_T=256, N_S=4, N_T=8, incep_ker=None, groups=8):
        super().__init__()
        if incep_ker is None:
            incep_ker = [3, 5, 7, 11]
        self.num_channels  = num_channels
        self.num_timesteps = num_timesteps
        shape_in = (num_timesteps, num_channels, 256, 256)
        self.simvp   = SimVP(shape_in, hid_S=hid_S, hid_T=hid_T,
                             N_S=N_S, N_T=N_T, incep_ker=incep_ker, groups=groups)
        self.readout = nn.Conv2d(num_timesteps * num_channels, n_horizons,
                                 kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        B, TC, H, W = x.shape
        x   = x.view(B, self.num_timesteps, self.num_channels, H, W)
        out = self.simvp(x)
        out = out.reshape(B, self.num_timesteps * self.num_channels, H, W)
        return self.readout(out).contiguous()


class EarthFormerWrapper(nn.Module):
    def __init__(self, num_channels, num_timesteps, n_horizons, model_cfg):
        super().__init__()
        self.num_channels  = num_channels
        self.num_timesteps = num_timesteps
        enc_depth      = list(model_cfg['enc_depth'])
        dec_depth      = list(model_cfg['dec_depth'])
        num_enc_blocks = len(enc_depth)
        num_dec_blocks = len(dec_depth)
        self.model = CuboidTransformerModel(
            input_shape  = tuple(model_cfg['input_shape']),
            target_shape = tuple(model_cfg['target_shape']),
            base_units   = model_cfg['base_units'],
            block_units  = model_cfg.get('block_units', None),
            scale_alpha  = model_cfg['scale_alpha'],
            enc_depth    = enc_depth,
            dec_depth    = dec_depth,
            enc_attn_patterns       = [model_cfg['self_pattern']]       * num_enc_blocks,
            dec_self_attn_patterns  = [model_cfg['cross_self_pattern']] * num_dec_blocks,
            dec_cross_attn_patterns = [model_cfg['cross_pattern']]      * num_dec_blocks,
            enc_cuboid_size          = [(4,4,4)]       * num_enc_blocks,
            enc_cuboid_strategy      = [('l','l','l')] * num_enc_blocks,
            enc_shift_size           = [(0,0,0)]        * num_enc_blocks,
            dec_self_cuboid_size     = [(4,4,4)]        * num_dec_blocks,
            dec_self_cuboid_strategy = [('l','l','l')]  * num_dec_blocks,
            dec_self_shift_size      = [(0,0,0)]         * num_dec_blocks,
            dec_cross_cuboid_hw      = [(4,4)]           * num_dec_blocks,
            dec_cross_cuboid_strategy= [('l','l')]       * num_dec_blocks,
            dec_cross_shift_hw       = [(0,0)]           * num_dec_blocks,
            dec_cross_n_temporal     = [2]               * num_dec_blocks,
            dec_cross_last_n_frames  = model_cfg['dec_cross_last_n_frames'],
            enc_use_inter_ffn          = model_cfg['enc_use_inter_ffn'],
            dec_use_inter_ffn          = model_cfg['dec_use_inter_ffn'],
            dec_hierarchical_pos_embed = model_cfg['dec_hierarchical_pos_embed'],
            dec_use_first_self_attn    = model_cfg['dec_use_first_self_attn'],
            dec_cross_start            = 0,
            num_heads  = model_cfg['num_heads'],
            attn_drop  = model_cfg['attn_drop'],
            proj_drop  = model_cfg['proj_drop'],
            ffn_drop   = model_cfg['ffn_drop'],
            downsample           = model_cfg['downsample'],
            downsample_type      = model_cfg['downsample_type'],
            upsample_type        = model_cfg['upsample_type'],
            upsample_kernel_size = model_cfg.get('upsample_kernel_size', 3),
            initial_downsample_type       = model_cfg['initial_downsample_type'],
            initial_downsample_activation = model_cfg['initial_downsample_activation'],
            initial_downsample_scale      = model_cfg.get('initial_downsample_scale', 1),
            initial_downsample_conv_layers  = model_cfg.get('initial_downsample_conv_layers', 2),
            final_upsample_conv_layers      = model_cfg.get('final_upsample_conv_layers', 2),
            initial_downsample_stack_conv_num_layers     = model_cfg['initial_downsample_stack_conv_num_layers'],
            initial_downsample_stack_conv_dim_list       = list(model_cfg['initial_downsample_stack_conv_dim_list']),
            initial_downsample_stack_conv_downscale_list = list(model_cfg['initial_downsample_stack_conv_downscale_list']),
            initial_downsample_stack_conv_num_conv_list  = list(model_cfg['initial_downsample_stack_conv_num_conv_list']),
            num_global_vectors     = model_cfg['num_global_vectors'],
            use_dec_self_global    = model_cfg['use_dec_self_global'],
            dec_self_update_global = model_cfg['dec_self_update_global'],
            use_dec_cross_global   = model_cfg['use_dec_cross_global'],
            use_global_vector_ffn  = model_cfg['use_global_vector_ffn'],
            use_global_self_attn   = model_cfg['use_global_self_attn'],
            separate_global_qkv    = model_cfg['separate_global_qkv'],
            global_dim_ratio       = model_cfg['global_dim_ratio'],
            z_init_method            = model_cfg['z_init_method'],
            ffn_activation           = model_cfg['ffn_activation'],
            gated_ffn                = model_cfg['gated_ffn'],
            norm_layer               = model_cfg['norm_layer'],
            padding_type             = model_cfg['padding_type'],
            pos_embed_type           = model_cfg['pos_embed_type'],
            use_relative_pos         = model_cfg['use_relative_pos'],
            self_attn_use_final_proj = model_cfg['self_attn_use_final_proj'],
            checkpoint_level         = model_cfg['checkpoint_level'],
            attn_linear_init_mode    = model_cfg['attn_linear_init_mode'],
            ffn_linear_init_mode     = model_cfg['ffn_linear_init_mode'],
            conv_init_mode           = model_cfg['conv_init_mode'],
            down_up_linear_init_mode = model_cfg['down_up_linear_init_mode'],
            norm_init_mode           = model_cfg['norm_init_mode'],
        )

    def forward(self, x):
        B, TC, H, W = x.shape
        x   = x.view(B, self.num_timesteps, self.num_channels, H, W)
        x   = rearrange(x, 'b t c h w -> b t h w c')
        out = self.model(x)
        out = out.squeeze(-1)
        return torch.clamp(out, -0.5, 1.5)


class VPTRWrapper(nn.Module):
    def __init__(self, num_channels, num_timesteps=5, n_horizons=4,
                 feat_dim=192, n_downsampling=3, encH=32, encW=32,
                 d_model=192, nhead=4, num_encoder_layers=2, num_decoder_layers=2,
                 dropout=0.1, window_size=4, Spatial_FFN_hidden_ratio=4, rpe=True):
        super().__init__()
        self.num_channels  = num_channels
        self.num_timesteps = num_timesteps
        self.n_horizons    = n_horizons
        self.encoder = VPTREnc(img_channels=num_channels, feat_dim=feat_dim,
                               n_downsampling=n_downsampling)
        self.former  = VPTRFormerNAR(num_past_frames=num_timesteps,
                                     num_future_frames=n_horizons,
                                     encH=encH, encW=encW, d_model=d_model,
                                     nhead=nhead, num_encoder_layers=num_encoder_layers,
                                     num_decoder_layers=num_decoder_layers,
                                     dropout=dropout, window_size=window_size,
                                     Spatial_FFN_hidden_ratio=Spatial_FFN_hidden_ratio,
                                     rpe=rpe)
        self.decoder  = VPTRDec(img_channels=num_channels, feat_dim=feat_dim,
                                n_downsampling=n_downsampling, out_layer='Sigmoid')
        self.out_proj = nn.Conv2d(num_channels, 1, kernel_size=1, bias=True)

    def forward(self, x):
        B, TC, H, W = x.shape
        x = x.view(B, self.num_timesteps, self.num_channels, H, W)
        past_feats  = self.encoder(x)
        pred_feats  = self.former(past_feats)
        pred_frames = self.decoder(pred_feats)
        B2, T2, C2, H2, W2 = pred_frames.shape
        out = self.out_proj(pred_frames.view(B2 * T2, C2, H2, W2))
        return out.view(B2, T2, H2, W2).contiguous()


def _strip_dp(sd):
    return {k.replace('module.', ''): v for k, v in sd.items()}


def load_all_models(cfg):
    n_ch  = 1
    n_hor = len(cfg.horizons)
    T     = cfg.num_in_frames
    models = {}

    ckpt = torch.load(cfg.smaat_checkpoint, map_location='cpu', weights_only=False)
    m = SmaAt_UNet(n_channels=T*n_ch, n_classes=n_hor,
                   kernels_per_layer=cfg.smaat_kernels_per_layer,
                   bilinear=cfg.smaat_bilinear,
                   reduction_ratio=cfg.smaat_reduction_ratio)
    m.load_state_dict(_strip_dp(ckpt['model_state_dict']), strict=False)
    models['smaat_unet'] = m.to(cfg.device).eval()

    ckpt = torch.load(cfg.convlstm_checkpoint, map_location='cpu', weights_only=False)
    m = ConvLSTMWrapper(num_channels=n_ch, num_timesteps=T, n_horizons=n_hor)
    m.load_state_dict(_strip_dp(ckpt['model_state_dict']), strict=False)
    models['convlstm'] = m.to(cfg.device).eval()

    ckpt = torch.load(cfg.simvp_checkpoint, map_location='cpu', weights_only=False)
    m = SimVPWrapper(num_channels=n_ch, num_timesteps=T, n_horizons=n_hor,
                     hid_S=cfg.hid_S, hid_T=cfg.hid_T, N_S=cfg.N_S, N_T=cfg.N_T,
                     incep_ker=cfg.incep_ker, groups=cfg.simvp_groups)
    m.load_state_dict(_strip_dp(ckpt['model_state_dict']), strict=False)
    models['simvp'] = m.to(cfg.device).eval()

    ckpt      = torch.load(cfg.earthformer_checkpoint, map_location='cpu', weights_only=False)
    ef_oc     = OmegaConf.load(cfg.earthformer_config_yml)
    model_cfg = OmegaConf.to_object(ef_oc.model)
    m = EarthFormerWrapper(num_channels=n_ch, num_timesteps=T,
                           n_horizons=n_hor, model_cfg=model_cfg)
    m.load_state_dict(_strip_dp(ckpt['model_state_dict']), strict=False)
    models['earthformer'] = m.to(cfg.device).eval()

    ckpt = torch.load(cfg.vptr_checkpoint, map_location='cpu', weights_only=False)
    m = VPTRWrapper(num_channels=n_ch, num_timesteps=T, n_horizons=n_hor,
                    feat_dim=cfg.vptr_feat_dim, n_downsampling=cfg.vptr_n_downsampling,
                    encH=cfg.vptr_encH, encW=cfg.vptr_encW, d_model=cfg.vptr_d_model,
                    nhead=cfg.vptr_nhead, num_encoder_layers=cfg.vptr_num_encoder_layers,
                    num_decoder_layers=cfg.vptr_num_decoder_layers,
                    dropout=cfg.vptr_dropout, window_size=cfg.vptr_window_size,
                    Spatial_FFN_hidden_ratio=cfg.vptr_spatial_ffn_ratio, rpe=cfg.vptr_rpe)
    m.load_state_dict(_strip_dp(ckpt['model_state_dict']), strict=False)
    models['vptr'] = m.to(cfg.device).eval()

    return models


def compute_ensemble(preds, horizons, ensemble_weights=None):
    if ensemble_weights is None:
        weights = {k: 1.0 for k in REAL_MODEL_KEYS}
    else:
        weights = ensemble_weights

    total_weight = sum(weights[k] for k in REAL_MODEL_KEYS)

    ensemble = {}
    for h in horizons:
        weighted_sum = sum(
            weights[k] * preds[k][h]
            for k in REAL_MODEL_KEYS
            if k in preds
        )
        n_available = sum(1 for k in REAL_MODEL_KEYS if k in preds)
        w_sum       = sum(weights[k] for k in REAL_MODEL_KEYS if k in preds)
        ensemble[h] = weighted_sum / w_sum

    return ensemble


def load_full_sample(row, cfg):
    dt            = datetime.strptime(row['datetime'], '%Y-%m-%d %H:%M:%S')
    history_times = get_history_times(dt, cfg.num_in_frames, cfg.stride_minutes)
    target_times  = get_target_times(dt, cfg.horizons)

    radar_planes = []
    for t in history_times:
        arr = np.load(radar_path(cfg.radar_base, t)).astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = softlog_transform(np.clip(arr, 0.0, CLIP_MAX_MMH))
        radar_planes.append(arr)

    input_np     = np.stack(radar_planes, axis=0).astype(np.float32)
    input_np     = np.nan_to_num(input_np, nan=0.0, posinf=0.0, neginf=0.0)
    input_tensor = torch.from_numpy(input_np).unsqueeze(0).float()

    targets_list, masks_list = [], []
    for t in target_times:
        arr  = np.load(radar_path(cfg.radar_base, t)).astype(np.float32)
        mask = np.isfinite(arr).astype(np.float32)
        arr  = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr  = softlog_transform(np.clip(arr, 0.0, CLIP_MAX_MMH))
        targets_list.append(arr)
        masks_list.append(mask)

    return input_tensor, np.stack(targets_list), np.stack(masks_list), dt


def detect_storms_two_level(rain_mm_h, operational_thr, extreme_thr,
                             min_pixels=10, disk_size=4,
                             min_area_km2=10.0, max_area_km2=100000.0,
                             pixel_area_km2=1.0):
    valid = np.isfinite(rain_mm_h)
    if valid.sum() == 0:
        empty = np.zeros_like(rain_mm_h, dtype=bool)
        return empty, empty, 0, 0
    mask_op = valid & (rain_mm_h >= operational_thr)
    if mask_op.sum() == 0:
        empty = np.zeros_like(rain_mm_h, dtype=bool)
        return empty, empty, 0, 0
    mask_op         = closing(mask_op, disk(disk_size))
    labeled_op, n_c = label(mask_op)
    final_op        = np.zeros_like(mask_op, dtype=bool)
    n_op            = 0
    for rid in range(1, n_c + 1):
        region = labeled_op == rid
        if min_area_km2 <= region.sum() * pixel_area_km2 <= max_area_km2:
            final_op[region] = True
            n_op += 1
    mask_ext = final_op & (rain_mm_h >= extreme_thr)
    if mask_ext.sum() == 0:
        return final_op, mask_ext, n_op, 0
    mask_ext = closing(mask_ext, disk(2))
    mask_ext = remove_small_objects(mask_ext, min_size=50)
    _, n_ext = label(mask_ext)
    return final_op, mask_ext, n_op, n_ext


def _render_image(ax, img_mm, storm_op, storm_ext, n_op, n_ext):
    ax.set_facecolor(BG_COLOR)
    im = ax.imshow(np.ma.masked_invalid(img_mm),
                   cmap=RAIN_CMAP, norm=RAIN_NORM, interpolation="nearest")
    if n_op > 0:
        ov = np.zeros((*storm_op.shape, 4))
        ov[storm_op] = [1.0, 0.6, 0.0, 0.25]
        ax.imshow(ov, interpolation="nearest")
    if n_ext > 0:
        ov = np.zeros((*storm_ext.shape, 4))
        ov[storm_ext] = [1.0, 0.0, 1.0, 0.35]
        ax.imshow(ov, interpolation="nearest")
    ax.axis("off")
    return im


def _build_title(kind, model_label, h, ts, p99, mae, n_op, n_ext, metrics_flag):
    if kind == 'gt':
        line1 = f"{model_label}  |  GT t+{h}min"
        line2 = (f"P99={p99:.1f}  Conv={n_op}  Ext={n_ext}"
                 if metrics_flag else ts.strftime('%Y-%m-%d %H:%M'))
    else:
        line1 = f"{model_label}  |  Pred t+{h}min"
        line2 = (f"P99={p99:.1f}  MAE={mae:.2f} mm/h"
                 if metrics_flag else ts.strftime('%Y-%m-%d %H:%M'))
    return f"{line1}\n{line2}"


def save_single_sample(sd, cfg, out_path, sample_idx):
    dt        = sd['dt']
    gt_mm     = sd['gt_mm']
    masks     = sd['masks']
    preds_all = sd['preds']
    horizons  = cfg.horizons
    n_hor     = len(horizons)
    metrics_flag = getattr(cfg, 'show_storm_metrics', False)

    N_IMG_COLS = n_hor * 2
    N_COLS     = N_IMG_COLS + 1
    n_rows     = len(MODELS)

    FIG_W = 4.2 * N_IMG_COLS + 1.0
    FIG_H = 3.8 * n_rows

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="#1a1a1a")

    gs = gridspec.GridSpec(
        n_rows, N_COLS, figure=fig,
        width_ratios=[1.0] * N_IMG_COLS + [0.03],
        hspace=0.10, wspace=0.04,
        left=0.02, right=0.97, top=0.97, bottom=0.03
    )

    fig.suptitle(
        f"Full Image Inference  |  Multi-Model Comparison  |  "
        f"Sample #{sample_idx+1}  |  {str(dt)[:16]}  |  P99={sd['p99']:.2f} mm/h",
        color="white", fontsize=12, fontweight="bold", y=0.995
    )

    last_im      = None
    target_times = {h: dt + timedelta(minutes=h) for h in horizons}

    for m_idx, minfo in enumerate(MODELS):
        mkey     = minfo['key']
        mlabel   = minfo['label']
        mcolor   = minfo['color']
        pred_mm  = preds_all.get(mkey, None)

        for h_idx, h in enumerate(horizons):
            valid = masks[h] > 0.5
            ts    = target_times[h]

            gt_data = np.where(valid & (gt_mm[h] >= 0.1), gt_mm[h], np.nan)
            gt_op, gt_ext, n_gt_op, n_gt_ext = detect_storms_two_level(
                np.where(valid, gt_mm[h], np.nan),
                cfg.operational_thr, cfg.extreme_thr,
                cfg.storm_min_pixels, cfg.morphology_disk_size,
                cfg.storm_min_area_km2, cfg.storm_max_area_km2, cfg.pixel_area_km2)
            gt_p99 = (np.percentile(gt_data[np.isfinite(gt_data)], 99)
                      if np.isfinite(gt_data).any() else 0.0)

            col_gt = h_idx * 2
            ax_gt  = fig.add_subplot(gs[m_idx, col_gt])
            im = _render_image(ax_gt, gt_data, gt_op, gt_ext, n_gt_op, n_gt_ext)
            last_im = im
            ax_gt.set_title(
                _build_title('gt', mlabel, h, ts,
                             gt_p99, None, n_gt_op, n_gt_ext, metrics_flag),
                color="white", fontsize=7, fontweight="bold", pad=3)
            for sp in ax_gt.spines.values():
                sp.set_edgecolor(mcolor); sp.set_linewidth(2.0)

            col_pred = h_idx * 2 + 1
            ax_pred  = fig.add_subplot(gs[m_idx, col_pred])

            if pred_mm is not None:
                pd_data = np.where(valid & (pred_mm[h] >= 0.1), pred_mm[h], np.nan)
                pr_op, pr_ext, n_pr_op, n_pr_ext = detect_storms_two_level(
                    np.where(valid, pred_mm[h], np.nan),
                    cfg.operational_thr, cfg.extreme_thr,
                    cfg.storm_min_pixels, cfg.morphology_disk_size,
                    cfg.storm_min_area_km2, cfg.storm_max_area_km2, cfg.pixel_area_km2)
                pd_p99 = (np.percentile(pd_data[np.isfinite(pd_data)], 99)
                          if np.isfinite(pd_data).any() else 0.0)
                mae = (float(np.abs(pred_mm[h][valid] - gt_mm[h][valid]).mean())
                       if valid.any() else float('nan'))
            else:
                pd_data  = np.full_like(gt_mm[h], np.nan)
                pr_op    = np.zeros_like(gt_mm[h], dtype=bool)
                pr_ext   = np.zeros_like(gt_mm[h], dtype=bool)
                n_pr_op  = n_pr_ext = 0
                pd_p99   = 0.0
                mae      = float('nan')

            _render_image(ax_pred, pd_data, pr_op, pr_ext, n_pr_op, n_pr_ext)
            ax_pred.set_title(
                _build_title('pred', mlabel, h, ts,
                             pd_p99, mae, n_pr_op, n_pr_ext, metrics_flag),
                color=mcolor, fontsize=7, fontweight="bold", pad=3)
            for sp in ax_pred.spines.values():
                sp.set_edgecolor(mcolor); sp.set_linewidth(2.0)

    if last_im is not None:
        cbar_ax = fig.add_subplot(gs[:, -1])
        cbar_ax.set_facecolor("#1a1a1a")
        cbar = fig.colorbar(last_im, cax=cbar_ax, ticks=RAIN_LEVELS,
                            spacing="proportional")
        cbar.outline.set_edgecolor("#555555"); cbar.outline.set_linewidth(0.5)
        cbar_ax.yaxis.set_ticks_position("left")
        cbar_ax.yaxis.set_label_position("left")
        cbar_ax.tick_params(axis="y", length=0, pad=4)
        plt.setp(cbar_ax.get_yticklabels(),
                 color="white", fontsize=7, fontfamily="monospace", fontweight="bold")
        cbar_ax_r = cbar_ax.twinx()
        cbar_ax_r.set_ylim(cbar_ax.get_ylim()); cbar_ax_r.set_yticks([])
        cbar_ax_r.set_ylabel("mm / h", color="#cccccc", fontsize=8,
                              fontweight="bold", labelpad=10,
                              rotation=270, va="bottom")
        cbar_ax_r.spines[:].set_visible(False)
        for thr, badge, col in [
            (cfg.operational_thr, f"OPR\n{cfg.operational_thr:.0f}", "#ffee00"),
            (cfg.extreme_thr,     f"EXT\n{cfg.extreme_thr:.0f}",     "#ff6600"),
        ]:
            cbar_ax.axhline(y=thr, color=col, linewidth=1.2,
                            linestyle="--", alpha=0.9,
                            xmin=-0.5, xmax=1.2, clip_on=False)
            cbar_ax.annotate(badge, xy=(1.15, thr),
                             xycoords=("axes fraction", "data"),
                             fontsize=6, fontweight="bold", color=col,
                             va="center", ha="left",
                             bbox=dict(boxstyle="round,pad=0.25",
                                       fc="#1a1a1a", ec=col, lw=0.8))

    legend_patches = [
        mpatches.Patch(facecolor=m['color'], label=m['label']) for m in MODELS
    ] + [
        mpatches.Patch(facecolor=(1.0, 0.6, 0.0, 0.4),
                       label=f"Operational >={cfg.operational_thr:.0f} mm/h"),
        mpatches.Patch(facecolor=(1.0, 0.0, 1.0, 0.5),
                       label=f"Extreme >={cfg.extreme_thr:.0f} mm/h"),
    ]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=len(legend_patches), fontsize=7, framealpha=0.3,
               facecolor="#333333", edgecolor="#555555", labelcolor="white",
               bbox_to_anchor=(0.48, 0.0))

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Full-image multi-model DL nowcasting comparison + ensemble (radar-only)")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--horizons", type=int, nargs="+", default=None)
    parser.add_argument("--metadata_csv", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--patch_overlap", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"])
    parser.add_argument("--show_storm_metrics", action="store_true")
    parser.add_argument("--rain_threshold", type=float, default=None)
    parser.add_argument("--csi_threshold_mmh", type=float, nargs="+", default=None)
    parser.add_argument("--mean_type", type=str, default="pooled", choices=["pooled", "simple"])
    args = parser.parse_args()

    cfg = InferenceConfig()

    if args.num_samples is not None:
        cfg.num_samples = args.num_samples
    if args.horizons is not None:
        cfg.horizons = args.horizons
    if args.metadata_csv is not None:
        cfg.metadata_csv = args.metadata_csv
    if args.out_dir is not None:
        cfg.out_dir = args.out_dir
    if args.patch_size is not None:
        cfg.patch_size = args.patch_size
    if args.patch_overlap is not None:
        cfg.patch_overlap = args.patch_overlap
    if args.device is not None:
        cfg.device = args.device
    if args.show_storm_metrics:
        cfg.show_storm_metrics = True
    if args.rain_threshold is not None:
        cfg.rain_threshold = args.rain_threshold
    if args.csi_threshold_mmh is not None:
        cfg.csi_threshold_mmh = args.csi_threshold_mmh

    cfg.detailed_metrics_csv = os.path.join(cfg.out_dir, "dl_fullimage_metrics.csv")
    cfg.mean_metrics_csv = os.path.join(cfg.out_dir, "dl_fullimage_mean_metrics.csv")
    cfg.pooled_mean_metrics_csv = os.path.join(cfg.out_dir, "dl_fullimage_pooled_mean_metrics.csv")
    cfg.simple_mean_metrics_csv = os.path.join(cfg.out_dir, "dl_fullimage_simple_mean_metrics.csv")

    os.makedirs(cfg.out_dir, exist_ok=True)

    df     = pd.read_csv(cfg.metadata_csv)
    df_top = df.nlargest(cfg.num_samples, 'p99(x_seq)').reset_index(drop=True)

    metrics_flag = getattr(cfg, 'show_storm_metrics', False)

    models = load_all_models(cfg)
    n_hor  = len(cfg.horizons)

    all_metrics = []

    for idx, row in df_top.iterrows():

        input_tensor, targets_np, masks_np, dt = load_full_sample(row, cfg)

        gt_mm = {}
        masks = {}
        for h_idx, h in enumerate(cfg.horizons):
            gt_mm[h] = inverse_transform_np(targets_np[h_idx])
            masks[h] = masks_np[h_idx]

        preds = {}
        for minfo in MODELS:
            mkey = minfo['key']
            if mkey == 'ensemble':
                continue

            mlabel = minfo['label']
            pred_cpu = run_model(models[mkey], input_tensor, n_hor, cfg)
            pred_np  = pred_cpu.squeeze(0).numpy()
            preds[mkey] = {h: inverse_transform_np(pred_np[h_idx])
                           for h_idx, h in enumerate(cfg.horizons)}

            maes = [
                f"t+{h}="
                f"{np.abs(preds[mkey][h][masks[h]>0.5] - gt_mm[h][masks[h]>0.5]).mean():.2f}"
                for h in cfg.horizons
                if masks[h].sum() > 0
            ]

        preds['ensemble'] = compute_ensemble(preds, cfg.horizons, cfg.ensemble_weights)
        ens_maes = [
            f"t+{h}="
            f"{np.abs(preds['ensemble'][h][masks[h]>0.5] - gt_mm[h][masks[h]>0.5]).mean():.2f}"
            for h in cfg.horizons
            if masks[h].sum() > 0
        ]

        for minfo in MODELS:
            mkey   = minfo['key']
            mlabel = minfo['label']
            for h in cfg.horizons:
                metrics = compute_metrics(preds[mkey][h], gt_mm[h], masks[h] > 0.5, cfg)
                if metrics is None:
                    continue
                row_result = {"model": mlabel, "datetime": dt.strftime("%Y-%m-%d %H:%M:%S"), "horizon_min": h}
                row_result.update(metrics)
                all_metrics.append(row_result)
        """
        #If we wart to save the results to a file
            dt = {
                'dt':    dt,
                'gt_mm': gt_mm,
                'masks': masks,
                'preds': preds,
                'p99':   float(row['p99(x_seq)']),
            }

            out_fname = f"sample_{idx+1:02d}_{str(dt)[:10]}_{str(dt)[11:16].replace(':','')}.png"
            out_path  = os.path.join(cfg.out_dir, out_fname)
            save_single_sample(sd, cfg, out_path, idx)
        """
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.to_csv(cfg.detailed_metrics_csv, index=False)

        pooled_df = compute_pooled_mean(metrics_df, cfg)
        pooled_df.to_csv(cfg.pooled_mean_metrics_csv, index=False)

        simple_df = compute_simple_mean(metrics_df, cfg)
        simple_df.to_csv(cfg.simple_mean_metrics_csv, index=False)

        (pooled_df if args.mean_type == "pooled" else simple_df).to_csv(
            cfg.mean_metrics_csv, index=False)


if __name__ == "__main__":
    main()