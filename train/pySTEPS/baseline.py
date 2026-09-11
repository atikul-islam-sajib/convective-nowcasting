import os
import json
from datetime import datetime, timedelta
import argparse
import multiprocessing as mp
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import distance_transform_edt
from pysteps import motion, nowcasts
import colors as PALETTE

CLIP_MAX_MMH = 128.0
RAIN_LEVELS = PALETTE.RAIN_LEVELS
RAIN_COLORS = PALETTE.RAIN_COLORS
RAIN_CMAP = PALETTE.RAIN_CMAP
RAIN_NORM = PALETTE.RAIN_NORM
BG_COLOR = PALETTE.BG_COLOR


class Config:
    categorical_thresholds = [5.0, 15.0]
    subset_size = None
    subset_manifest_csv = os.path.join('MEAN_PYSTEPS', 'final_test_manifest.csv')
    use_existing_subset_manifest = True 

    out_dir = 'LK_EXTRAPOLATION_PYSTEPS_FINAL'
    num_processes = max(1, (os.cpu_count() or 4) - 1)
    radar_base = '/home/fe/sajib/scratch/weather-data/radar_de'
    sat_base = '/home/fe/sajib/scratch/weather-data/satellite_de_regridded'
    channels = ['CH7', 'CH9']
    metadata_csv = 'metadata_patch/test_fullimage_summer.csv'
    channel_stats_json = 'metadata/channel_stats.json'
    detailed_metrics_csv = os.path.join(out_dir, 'lk_extrapolation_radar_multimodal_metrics.csv')
    mean_metrics_csv = os.path.join(out_dir, 'lk_extrapolation_radar_multimodal_mean_metrics.csv')
    num_in_frames = 4 
    lk_num_frames = 4 

    stride_minutes = 5
    horizons = [15, 30, 45, 60]
    metric_horizons = [15, 30, 45, 60]
    rain_threshold = 0.1
    precip_thr_db = 10.0 * np.log10(rain_threshold)
    zerovalue_db = -15.0
    extrap_method = 'semilagrangian'
    motion_method = 'LK'
    nowcast_method = 'extrapolation' 
    sprog_ar_order = 2  
    sprog_n_cascade_levels = 6
    extrap_interp_order = 1 
    save_motion_diagnostics = False
    save_overlay_diagnostics = False 
    overlay_diagnostic_horizons = (15, 30)
    use_persistence_blend = False  
    persistence_blend_max_horizon = 30 
    use_active_rain_subset = True
    active_rain_threshold_mmh = 1.0
    subset_seed = 42
    use_existing_subset_manifest = True
    num_samples = None
    select_top_p99 = True
    select_on_the_hour = False
    csi_threshold_mmh = 15.0  
    psnr_data_range = CLIP_MAX_MMH
    save_images = False
    save_images_only_datetimes = None
    operational_thr = 15
    extreme_thr = 40


def radar_path(base, dt):
    return os.path.join(base, dt.strftime("%y%m"), dt.strftime("%d"), dt.strftime("%y%m%d_%H%M") + '.npy')


def satellite_path(base, dt, channel):
    return os.path.join(base, dt.strftime("%Y"), dt.strftime("%m"), dt.strftime("%d"), dt.strftime("%H%M%S") + f'_{channel}.npy')


def get_history_times(current_time, num_frames, stride_minutes):
    start = current_time - (num_frames - 1) * timedelta(minutes=stride_minutes)
    return [start + timedelta(minutes=stride_minutes * i) for i in range(num_frames)]


def get_target_times(current_time, horizons):
    return [current_time + timedelta(minutes=h) for h in horizons]


def load_npy(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f'\nFile not found:\n{path}')
    return np.load(path).astype(np.float32)


def load_channel_stats(cfg):
    if not os.path.exists(cfg.channel_stats_json):
        print('\nWARNING: channel_stats.json not found.')
        print('Satellite normalization will use per-sequence robust normalization.')
        return None
    with open(cfg.channel_stats_json, 'r') as f:
        stats = json.load(f)
    return {int(k): v for k, v in stats.items()}

def inpaint_nearest(arr, valid_mask):
    if valid_mask.all():
        return arr
    if not valid_mask.any():
        return np.zeros_like(arr)
    idx = distance_transform_edt(~valid_mask, return_distances=False, return_indices=True)
    return arr[tuple(idx)]


def load_sample(row, cfg):
    dt = datetime.strptime(row['datetime'], '%Y-%m-%d %H:%M:%S')
    history_times = get_history_times(dt, cfg.lk_num_frames, cfg.stride_minutes)
    target_times = get_target_times(dt, cfg.metric_horizons)

    radar_history = []
    radar_history_masks = []
    for t in history_times:
        path = radar_path(cfg.radar_base, t)
        arr = load_npy(path)
        valid = np.isfinite(arr)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.clip(arr, 0.0, CLIP_MAX_MMH)
        arr = inpaint_nearest(arr, valid)
        radar_history.append(arr)
        radar_history_masks.append(valid)
    radar_history = np.stack(radar_history, axis=0).astype(np.float32)
    radar_history_masks = np.stack(radar_history_masks, axis=0)

    satellite_history = {}
    for channel in cfg.channels:
        channel_frames = []
        for t in history_times:
            path = satellite_path(cfg.sat_base, t, channel)
            arr = load_npy(path)
            valid = np.isfinite(arr)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            arr = inpaint_nearest(arr, valid)
            channel_frames.append(arr)
        satellite_history[channel] = np.stack(channel_frames, axis=0).astype(np.float32)

    gt = {}
    masks = {}
    for h, t in zip(cfg.metric_horizons, target_times):
        path = radar_path(cfg.radar_base, t)
        arr = load_npy(path)
        valid = np.isfinite(arr)
        clean = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        clean = np.clip(clean, 0.0, CLIP_MAX_MMH)
        gt[h] = clean.astype(np.float32)
        masks[h] = valid 

    return (radar_history, radar_history_masks, satellite_history, gt, masks, dt, history_times, target_times)


def rainrate_to_db(rain, threshold=0.1, zerovalue=-15.0):
    rain = np.asarray(rain, dtype=np.float32)
    result = np.full_like(rain, zerovalue, dtype=np.float32)
    wet = rain >= threshold
    result[wet] = 10.0 * np.log10(rain[wet])
    return result


def db_to_rainrate(data, cfg):
    data = np.asarray(data, dtype=np.float32)
    rain = np.zeros_like(data, dtype=np.float32)
    valid = np.isfinite(data) & (data > cfg.zerovalue_db)
    rain[valid] = 10.0 ** (data[valid] / 10.0)
    rain[rain < cfg.rain_threshold] = 0.0
    return np.clip(rain, 0.0, CLIP_MAX_MMH).astype(np.float32)


def normalize_satellite_sequence(seq, channel, channel_stats):
    seq = np.asarray(seq, dtype=np.float32)
    ch_number = int(channel.replace('CH', ''))
    if channel_stats is not None and ch_number in channel_stats:
        mean = float(channel_stats[ch_number]['mean'])
        std = float(channel_stats[ch_number]['std'])
        seq = (seq - mean) / (std + 1e-08)
    else:
        finite = np.isfinite(seq)
        if not np.any(finite):
            return np.zeros_like(seq)
        median = np.median(seq[finite])
        p25 = np.percentile(seq[finite], 25)
        p75 = np.percentile(seq[finite], 75)
        scale = max(p75 - p25, 1e-06)
        seq = (seq - median) / scale
    seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
    low = np.percentile(seq, 1)
    high = np.percentile(seq, 99)
    if high - low < 1e-08:
        return np.zeros_like(seq)
    seq = (seq - low) / (high - low)
    seq = np.clip(seq, 0.0, 1.0)
    return seq.astype(np.float32)


def estimate_lk_motion(frames, name, cfg):
    frames = np.asarray(frames, dtype=np.float32)
    frames = np.nan_to_num(frames, nan=0.0, posinf=0.0, neginf=0.0)
    if cfg.motion_method in ('VET', 'proesmans'):
        motion_frames = frames[-3:] if frames.shape[0] >= 3 else frames[-2:]
    else:
        motion_frames = frames

    print(f'       {cfg.motion_method} motion: {name}  (using last {motion_frames.shape[0]} of {frames.shape[0]} frames)')
    oflow = motion.get_method(cfg.motion_method)
    velocity = oflow(motion_frames)
    velocity = np.nan_to_num(velocity, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    mean_speed = np.mean(np.sqrt(velocity[0] ** 2 + velocity[1] ** 2))
    print(f'          mean speed = {mean_speed:.4f} pixels/timestep')
    return velocity


def estimate_radar_motion(radar_db, cfg):
    return estimate_lk_motion(radar_db, 'Radar', cfg)


def estimate_multimodal_motion(radar_db, satellite_history, channel_stats, cfg):
    v_radar = estimate_lk_motion(radar_db, 'Radar', cfg)
    velocities = [v_radar]
    for channel in cfg.channels:
        sat_seq = satellite_history[channel][-cfg.lk_num_frames:]
        sat_normalized = normalize_satellite_sequence(sat_seq, channel, channel_stats)
        v_sat = estimate_lk_motion(sat_normalized, channel, cfg)
        velocities.append(v_sat)
    velocity_stack = np.stack(velocities, axis=0)
    multimodal_velocity = np.median(velocity_stack, axis=0)
    multimodal_velocity = np.nan_to_num(multimodal_velocity, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    mean_speed = np.mean(np.sqrt(multimodal_velocity[0] ** 2 + multimodal_velocity[1] ** 2))
    print(f'       Combined multimodal mean speed = {mean_speed:.4f}')
    return multimodal_velocity


def run_lk_extrapolation_with_velocity(radar_history, velocity, cfg):
    latest_radar = radar_history[-1]
    n_steps = max(cfg.metric_horizons) // cfg.stride_minutes

    if cfg.nowcast_method == 'sprog':
        n_needed = cfg.sprog_ar_order + 1
        precip_stack = radar_history[-n_needed:]
        precip_stack_db = rainrate_to_db(precip_stack, threshold=cfg.rain_threshold, zerovalue=cfg.zerovalue_db)
        precip_stack_db = np.nan_to_num(
            precip_stack_db, nan=cfg.zerovalue_db, posinf=cfg.zerovalue_db, neginf=cfg.zerovalue_db
        ).astype(np.float32)
        sprog = nowcasts.get_method('sprog')
        sprog_kwargs = dict(
            precip_thr=cfg.precip_thr_db,
            n_cascade_levels=cfg.sprog_n_cascade_levels,
            extrap_method=cfg.extrap_method,
            extrap_kwargs={'interp_order': cfg.extrap_interp_order},
        )
        ar_orders_to_try = sorted({cfg.sprog_ar_order, 2}, reverse=True)
        forecast_db = None
        last_error = None
        for attempt_ar_order in ar_orders_to_try:
            n_needed_attempt = attempt_ar_order + 1
            precip_stack_attempt = radar_history[-n_needed_attempt:]
            precip_stack_db_attempt = rainrate_to_db(precip_stack_attempt, threshold=cfg.rain_threshold, zerovalue=cfg.zerovalue_db)
            precip_stack_db_attempt = np.nan_to_num(
                precip_stack_db_attempt, nan=cfg.zerovalue_db, posinf=cfg.zerovalue_db, neginf=cfg.zerovalue_db
            ).astype(np.float32)
            try:
                forecast_db = sprog(precip_stack_db_attempt, velocity, n_steps, ar_order=attempt_ar_order, **sprog_kwargs)
                if attempt_ar_order != cfg.sprog_ar_order:
                    print(f'    [WARN] sprog ar_order={cfg.sprog_ar_order} failed, succeeded at ar_order={attempt_ar_order}')
                break
            except Exception as error:
                last_error = error
                print(f'    [WARN] sprog ar_order={attempt_ar_order} failed: {error}')
        if forecast_db is None:
            print('    [WARN] sprog failed at all ar_order attempts -- falling back to plain extrapolation for this sample')
            latest_radar_db = rainrate_to_db(latest_radar, threshold=cfg.rain_threshold, zerovalue=cfg.zerovalue_db)
            latest_radar_db = np.nan_to_num(
                latest_radar_db, nan=cfg.zerovalue_db, posinf=cfg.zerovalue_db, neginf=cfg.zerovalue_db
            ).astype(np.float32)
            extrapolate = nowcasts.get_method('extrapolation')
            forecast_db = extrapolate(
                latest_radar_db, velocity, n_steps,
                extrap_method=cfg.extrap_method,
                extrap_kwargs={'interp_order': cfg.extrap_interp_order},
            )
    else:
        latest_radar_db = rainrate_to_db(
            latest_radar,
            threshold=cfg.rain_threshold,
            zerovalue=cfg.zerovalue_db,
        )
        latest_radar_db = np.nan_to_num(
            latest_radar_db,
            nan=cfg.zerovalue_db,
            posinf=cfg.zerovalue_db,
            neginf=cfg.zerovalue_db,
        ).astype(np.float32)
        extrapolate = nowcasts.get_method('extrapolation')
        forecast_db = extrapolate(
            latest_radar_db,
            velocity,
            n_steps,
            extrap_method=cfg.extrap_method,
            extrap_kwargs={'interp_order': cfg.extrap_interp_order},
        )

    forecast_db = np.asarray(forecast_db, dtype=np.float32)
    forecast_mm = db_to_rainrate(forecast_db, cfg)

    predictions = {}
    for horizon in cfg.metric_horizons:
        step = horizon // cfg.stride_minutes
        index = step - 1
        pred = np.clip(forecast_mm[index], 0.0, CLIP_MAX_MMH).astype(np.float32)
        if cfg.use_persistence_blend and horizon <= cfg.persistence_blend_max_horizon:
            w = 1.0 - (horizon / cfg.persistence_blend_max_horizon)
            pred = np.clip((1.0 - w) * pred + w * latest_radar, 0.0, CLIP_MAX_MMH).astype(np.float32)

        predictions[horizon] = pred

    return predictions


def run_both_baselines(radar_history, satellite_history, channel_stats, cfg):
    print(f'\n    Preparing {cfg.motion_method} extrapolation input...')
    radar_recent = radar_history[-cfg.lk_num_frames:]
    radar_db = rainrate_to_db(radar_recent, threshold=cfg.rain_threshold, zerovalue=cfg.zerovalue_db)
    print(f'\n    [1/2] pySTEPS RADAR ({cfg.motion_method})')
    radar_velocity = estimate_radar_motion(radar_db, cfg)
    radar_predictions = run_lk_extrapolation_with_velocity(radar_history, radar_velocity, cfg)
    print(f'\n    [2/2] pySTEPS MULTIMODAL (Radar + CH7 + CH9 motion, {cfg.motion_method})')
    multimodal_velocity = estimate_multimodal_motion(radar_db, satellite_history, channel_stats, cfg)
    multimodal_predictions = run_lk_extrapolation_with_velocity(radar_history, multimodal_velocity, cfg)
    return (radar_predictions, multimodal_predictions, radar_velocity, multimodal_velocity)

def save_motion_diagnostic(t0_frame, velocity, name, dt, cfg, sample_index, step=12):
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
    ax.set_facecolor(BG_COLOR)
    display_image = np.ma.masked_less(t0_frame, 0.1)
    ax.imshow(display_image, cmap=RAIN_CMAP, norm=RAIN_NORM, interpolation='nearest')
    h, w = t0_frame.shape
    ys, xs = np.mgrid[0:h:step, 0:w:step]
    vx = velocity[0][::step, ::step]
    vy = velocity[1][::step, ::step]
    ax.quiver(xs, ys, vx, vy, color='cyan', scale=None, width=0.002, alpha=0.85)
    ax.set_title(f'{name} velocity field ({cfg.motion_method}) — {dt.strftime("%Y-%m-%d %H:%M")}', fontsize=10, fontweight='bold')
    ax.axis('off')
    out_path = os.path.join(cfg.out_dir, f'motion_diag_sample{sample_index:03d}_{name.lower()}_{cfg.motion_method}.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'    Motion diagnostic saved: {out_path}')

def save_overlay_diagnostic(prediction, target, mask, dt, cfg, sample_index, model_name, horizon):
    fig, ax = plt.subplots(figsize=(9, 9), facecolor='white')
    ax.set_facecolor(BG_COLOR)
    display_image = np.ma.masked_less(prediction, 0.1)
    display_image = np.ma.masked_where(~mask, display_image)
    ax.imshow(display_image, cmap=RAIN_CMAP, norm=RAIN_NORM, interpolation='nearest')
    gt_event = np.where(mask, (target >= cfg.csi_threshold_mmh).astype(np.float32), 0.0)
    ax.contour(gt_event, levels=[0.5], colors='black', linewidths=1.5)
    ax.set_title(
        f'{model_name} t+{horizon}min vs GT {cfg.csi_threshold_mmh:.0f}mm/h boundary (black) — {dt.strftime("%Y-%m-%d %H:%M")}',
        fontsize=10, fontweight='bold',
    )
    ax.axis('off')
    out_path = os.path.join(cfg.out_dir, f'overlay_diag_sample{sample_index:03d}_{model_name.lower()}_t{horizon}.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'    Overlay diagnostic saved: {out_path}')


def contingency_counts(pred, target, threshold):
    pred_event = pred >= threshold
    target_event = target >= threshold
    hits = np.sum(pred_event & target_event)
    misses = np.sum(~pred_event & target_event)
    false_alarms = np.sum(pred_event & ~target_event)
    correct_negatives = np.sum(~pred_event & ~target_event)
    return (float(hits), float(misses), float(false_alarms), float(correct_negatives))


def csi_from_counts(hits, misses, false_alarms):
    denom = hits + misses + false_alarms
    if denom == 0:
        return np.nan
    return hits / denom


def ets_from_counts(hits, misses, false_alarms, correct_negatives):
    total = hits + misses + false_alarms + correct_negatives
    if total == 0:
        return np.nan
    random_hits = (hits + misses) * (hits + false_alarms) / total
    denom = hits + misses + false_alarms - random_hits
    if denom == 0:
        return np.nan
    return (hits - random_hits) / denom


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
    psnr = compute_psnr(p, y, cfg.psnr_data_range)

    result = {'MAE': float(mae), 'MSE': float(mse), 'PSNR': float(psnr)}

    for threshold in cfg.categorical_thresholds:
        suffix = str(int(threshold)) if float(threshold).is_integer() else str(threshold)
        h, m, fa, cn = contingency_counts(p, y, threshold)
        result[f'THR{suffix}_hits'] = h
        result[f'THR{suffix}_misses'] = m
        result[f'THR{suffix}_fa'] = fa
        result[f'THR{suffix}_cn'] = cn

    return result


def detect_storms_two_level(rain_mm_h, operational_thr, extreme_thr, min_pixels=200, disk_size=4, min_area_km2=200.0, max_area_km2=10000.0, pixel_area_km2=1.0):
    from skimage.morphology import closing, remove_small_objects, disk as sk_disk
    from scipy.ndimage import label as sp_label
    valid = np.isfinite(rain_mm_h)
    if valid.sum() == 0:
        empty = np.zeros_like(rain_mm_h, dtype=bool)
        return (empty, empty, 0, 0)
    mask_op = valid & (rain_mm_h >= operational_thr)
    if mask_op.sum() == 0:
        empty = np.zeros_like(rain_mm_h, dtype=bool)
        return (empty, empty, 0, 0)
    mask_op = closing(mask_op, sk_disk(disk_size))
    labeled_op, n_c = sp_label(mask_op)
    final_op = np.zeros_like(mask_op, dtype=bool)
    n_op = 0
    for rid in range(1, n_c + 1):
        region = labeled_op == rid
        if min_area_km2 <= region.sum() * pixel_area_km2 <= max_area_km2:
            final_op[region] = True
            n_op += 1
    mask_ext = final_op & (rain_mm_h >= extreme_thr)
    if mask_ext.sum() == 0:
        return (final_op, mask_ext, n_op, 0)
    mask_ext = closing(mask_ext, sk_disk(2))
    mask_ext = remove_small_objects(mask_ext, min_size=50)
    _, n_ext = sp_label(mask_ext)
    return (final_op, mask_ext, n_op, n_ext)


def render_precip(ax, image, valid_mask=None, operational_thr=None, extreme_thr=None):
    ax.set_facecolor(BG_COLOR)
    display_image = np.ma.masked_less(image, 0.1)
    if valid_mask is not None:
        display_image = np.ma.masked_where(~valid_mask, display_image)
    im = ax.imshow(display_image, cmap=RAIN_CMAP, norm=RAIN_NORM, interpolation='nearest')
    if operational_thr is not None and extreme_thr is not None:
        field = np.where(valid_mask, image, np.nan) if valid_mask is not None else image
        op, ext, n_op, n_ext = detect_storms_two_level(field, operational_thr, extreme_thr)
        if n_op > 0:
            ov = np.zeros((*op.shape, 4))
            ov[op] = PALETTE.OP_OVERLAY_RGBA
            ax.imshow(ov, interpolation='nearest')
        if n_ext > 0:
            ov = np.zeros((*ext.shape, 4))
            ov[ext] = PALETTE.EXT_OVERLAY_RGBA
            ax.imshow(ov, interpolation='nearest')
    ax.axis('off')
    return im


def save_comparison_figure(gt, radar_preds, multimodal_preds, masks, dt, cfg, sample_index, t0_frame=None, t0_mask=None, horizons_to_show=None, filename_suffix=''):
    RADAR_COLOR = PALETTE.RADAR_COLOR
    MULTIMODAL_COLOR = PALETTE.MULTIMODAL_COLOR
    GT_COLOR = PALETTE.GT_COLOR
    fig = plt.figure(figsize=(20, 12), facecolor='white')
    show_t0 = t0_frame is not None
    base_horizons = horizons_to_show if horizons_to_show is not None else cfg.horizons
    display_horizons = ([0] if show_t0 else []) + list(base_horizons)
    n_cols = len(display_horizons)
    gs = fig.add_gridspec(3, n_cols + 1, width_ratios=[1] * n_cols + [0.055], left=0.1, right=0.94, bottom=0.08, top=0.86, hspace=0.2, wspace=0.05)
    last_im = None
    for col, h in enumerate(display_horizons):
        ax = fig.add_subplot(gs[0, col])
        if h == 0:
            panel_img = t0_frame
            panel_mask = t0_mask
        else:
            panel_img = gt[h]
            panel_mask = masks[h]
        last_im = render_precip(ax, panel_img, valid_mask=panel_mask, operational_thr=cfg.operational_thr, extreme_thr=cfg.extreme_thr)
        target_time = dt + timedelta(minutes=h)
        if h == 0:
            title_text = 'Ground Truth  |  Analysis (t+0)\n' + target_time.strftime("%Y-%m-%d %H:%M")
        else:
            title_text = f'Ground Truth  |  GT t+{h}min\n' + target_time.strftime("%Y-%m-%d %H:%M")
        ax.set_title(title_text, fontsize=9, fontweight='bold', pad=10)
    for col, h in enumerate(display_horizons):
        ax = fig.add_subplot(gs[1, col])
        if h == 0:
            panel_img = t0_frame
            panel_mask = t0_mask
        else:
            panel_img = radar_preds[h]
            panel_mask = masks[h]
        render_precip(ax, panel_img, valid_mask=panel_mask, operational_thr=cfg.operational_thr, extreme_thr=cfg.extreme_thr)
        target_time = dt + timedelta(minutes=h)
        if h == 0:
            title_text = 'PySteps Radar  |  Analysis (t+0)\n' + target_time.strftime("%Y-%m-%d %H:%M")
        else:
            title_text = f'PySteps Radar  |  Pred t+{h}min\n' + target_time.strftime("%Y-%m-%d %H:%M")
        ax.set_title(title_text, fontsize=9, fontweight='bold', color=RADAR_COLOR, pad=10)
    for col, h in enumerate(display_horizons):
        ax = fig.add_subplot(gs[2, col])
        if h == 0:
            panel_img = t0_frame
            panel_mask = t0_mask
        else:
            panel_img = multimodal_preds[h]
            panel_mask = masks[h]
        render_precip(ax, panel_img, valid_mask=panel_mask, operational_thr=cfg.operational_thr, extreme_thr=cfg.extreme_thr)
        target_time = dt + timedelta(minutes=h)
        if h == 0:
            title_text = 'PySteps Multimodal  |  Analysis (t+0)\n' + target_time.strftime("%Y-%m-%d %H:%M")
        else:
            title_text = f'PySteps Multimodal  |  Pred t+{h}min\n' + target_time.strftime("%Y-%m-%d %H:%M")
        ax.set_title(title_text, fontsize=9, fontweight='bold', color=MULTIMODAL_COLOR, pad=10)
    fig.text(0.055, 0.76, 'Ground\nTruth', ha='center', va='center', fontsize=14, fontweight='bold', color=GT_COLOR)
    fig.text(0.055, 0.5, 'PySteps\nRadar', ha='center', va='center', fontsize=14, fontweight='bold', color=RADAR_COLOR)
    fig.text(0.055, 0.245, 'PySteps\nMultimodal', ha='center', va='center', fontsize=14, fontweight='bold', color=MULTIMODAL_COLOR)
    cbar_ax = fig.add_subplot(gs[:, n_cols])
    cbar = fig.colorbar(last_im, cax=cbar_ax, ticks=RAIN_LEVELS, spacing='proportional')
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label('mm / h', fontsize=10, fontweight='bold')
    cbar_ax.axhline(y=cfg.operational_thr, linestyle='--', linewidth=1.3, color='#f0a000')
    cbar_ax.annotate(f'OPR\n{cfg.operational_thr:.0f}', xy=(1.15, cfg.operational_thr), xycoords=('axes fraction', 'data'), fontsize=7, color='#d99200', fontweight='bold', va='center', bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#d99200', lw=0.8))
    cbar_ax.axhline(y=cfg.extreme_thr, linestyle='--', linewidth=1.3, color='#dd5500')
    cbar_ax.annotate(f'EXT\n{cfg.extreme_thr:.0f}', xy=(1.15, cfg.extreme_thr), xycoords=('axes fraction', 'data'), fontsize=7, color='#dd5500', fontweight='bold', va='center', bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#dd5500', lw=0.8))
    fig.suptitle('Comparison of Ground Truth and Lucas-Kanade Extrapolation Predictions', fontsize=17, fontweight='bold', y=0.98)
    handles = [mpatches.Patch(color=GT_COLOR, label='Ground Truth'), mpatches.Patch(color=RADAR_COLOR, label='PySteps Radar'), mpatches.Patch(color=MULTIMODAL_COLOR, label='PySteps Multimodal (Radar+CH7+CH9)'), mpatches.Patch(color=PALETTE.OP_LEGEND_RGBA, label=f'Operational >={cfg.operational_thr:.0f} mm/h'), mpatches.Patch(color=PALETTE.EXT_LEGEND_RGBA, label=f'Extreme >={cfg.extreme_thr:.0f} mm/h')]
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=8, frameon=True, bbox_to_anchor=(0.52, 0.025))
    output_path = os.path.join(cfg.out_dir, f'pysteps_lk_extrapolation_comparison_sample{sample_index:03d}_{dt.strftime("%Y-%m-%d")}{filename_suffix}.png')
    fig.savefig(output_path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'\n    Figure saved:\n    {output_path}')


def process_one_sample(args):
    sample_idx, row, cfg, channel_stats = args
    results = []
    print()
    print('=' * 80)
    print(f'SAMPLE {sample_idx + 1}')
    print(f'Datetime: {row["datetime"]}')
    print('=' * 80)
    try:
        radar_history, radar_history_masks, satellite_history, gt, masks, dt, history_times, target_times = load_sample(row, cfg)
    except Exception as error:
        print(f'\nLOAD ERROR:\n{error}')
        return results
    print('\n    Historical observations:')
    for t in history_times:
        print(f'       {t.strftime("%Y-%m-%d %H:%M")}')
    print('\n    LK extrapolation motion input:')
    for t in history_times[-cfg.lk_num_frames:]:
        print(f'       {t.strftime("%Y-%m-%d %H:%M")}')
    print('\n    Targets:')
    for horizon, t in zip(cfg.metric_horizons, target_times):
        print(f'       +{horizon:02d}: {t.strftime("%Y-%m-%d %H:%M")}')
    print(f'\n    Radar shape : {radar_history.shape}')
    for ch in cfg.channels:
        print(f'    {ch} shape   : {satellite_history[ch].shape}')
    radar_hw = radar_history.shape[-2:]
    for ch in cfg.channels:
        sat_hw = satellite_history[ch].shape[-2:]
        if sat_hw != radar_hw:
            print(f'\nSHAPE ERROR: {ch} grid {sat_hw} does not match radar grid {radar_hw}. Use the regridded satellite data.')
            return results
    try:
        radar_predictions, multimodal_predictions, radar_velocity, multimodal_velocity = run_both_baselines(radar_history, satellite_history, channel_stats, cfg)
    except Exception as error:
        print(f'\nFORECAST ERROR:\n{error}')
        return results
    if cfg.save_motion_diagnostics:
        save_motion_diagnostic(radar_history[-1], radar_velocity, 'Radar', dt, cfg, sample_idx + 1)
        save_motion_diagnostic(radar_history[-1], multimodal_velocity, 'Multimodal', dt, cfg, sample_idx + 1)
    if cfg.save_overlay_diagnostics:
        for model_name, predictions in [('PySteps_Radar', radar_predictions), ('PySteps_Multimodal', multimodal_predictions)]:
            for horizon in cfg.overlay_diagnostic_horizons:
                if horizon in predictions and horizon in gt:
                    save_overlay_diagnostic(predictions[horizon], gt[horizon], masks[horizon], dt, cfg, sample_idx + 1, model_name, horizon)
    print('\n' + '-' * 80)
    print('METRICS')
    print('-' * 80)
    for model_name, predictions in [('PySteps_Radar', radar_predictions), ('PySteps_Multimodal', multimodal_predictions)]:
        print(f'\n    {model_name}')
        for horizon in cfg.metric_horizons:
            metrics = compute_metrics(predictions[horizon], gt[horizon], masks[horizon], cfg)
            if metrics is None:
                continue
            row_result = {'model': model_name, 'datetime': dt.strftime("%Y-%m-%d %H:%M:%S"), 'horizon_min': horizon}
            row_result.update(metrics)
            results.append(row_result)
            thr_parts = []
            for threshold in cfg.categorical_thresholds:
                suffix = str(int(threshold)) if float(threshold).is_integer() else str(threshold)
                h = metrics[f'THR{suffix}_hits']
                m = metrics[f'THR{suffix}_misses']
                fa = metrics[f'THR{suffix}_fa']
                cn = metrics[f'THR{suffix}_cn']
                csi = csi_from_counts(h, m, fa)
                ets = ets_from_counts(h, m, fa, cn)
                thr_parts.append(f'CSI@{suffix}={csi:.4f} ETS@{suffix}={ets:.4f}')
            thr_str = ' | '.join(thr_parts)
            print(f'       t+{horizon:02d} | MAE={metrics["MAE"]:.4f} | MSE={metrics["MSE"]:.4f} | PSNR={metrics["PSNR"]:.4f} | {thr_str}')
    should_save_image = cfg.save_images and (
        cfg.save_images_only_datetimes is None or row['datetime'] in cfg.save_images_only_datetimes
    )
    if should_save_image:
        save_comparison_figure(gt, radar_predictions, multimodal_predictions, masks, dt, cfg, sample_idx + 1, t0_frame=None, t0_mask=None, horizons_to_show=cfg.metric_horizons)
    return results

def compute_pooled_mean(metrics_df, cfg):
    rows = []
    for (model, horizon), g in metrics_df.groupby(['model', 'horizon_min']):
        row = {'model': model, 'horizon_min': horizon}
        row['MAE'] = g['MAE'].mean()
        row['MSE'] = g['MSE'].mean()
        row['PSNR'] = g['PSNR'].mean()

        csi_vals, ets_vals = [], []
        for threshold in cfg.categorical_thresholds:
            suffix = str(int(threshold)) if float(threshold).is_integer() else str(threshold)
            h = g[f'THR{suffix}_hits'].sum()
            m = g[f'THR{suffix}_misses'].sum()
            fa = g[f'THR{suffix}_fa'].sum()
            cn = g[f'THR{suffix}_cn'].sum()
            csi_vals.append(csi_from_counts(h, m, fa))
            ets_vals.append(ets_from_counts(h, m, fa, cn))

        row['CSI'] = float(np.nanmean(csi_vals)) if csi_vals else float('nan')
        row['ETS'] = float(np.nanmean(ets_vals)) if ets_vals else float('nan')
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description='pySTEPS Radar + Multimodal Lucas-Kanade extrapolation baseline')
    parser.add_argument('--num_samples', type=int, default=None, help=f'LEGACY MODE ONLY (Config.use_active_rain_subset=False). Number of samples to evaluate (default: Config.num_samples = {Config.num_samples!r}). Pass 0 to run the FULL filtered test set (every event passing select_on_the_hour/select_top_p99, equivalent to Config.num_samples=None) instead of a fixed count. Example: --num_samples 50')
    parser.add_argument('--subset_size', type=int, default=None, help=f"How many active-rain sequences to evaluate (default: Config.subset_size = {Config.subset_size}). 1,000-2,000 is typically enough for CSI/ETS/MAE to converge while keeping LK extrapolation runtime tractable. Example: --subset_size 1500")
    parser.add_argument('--active_rain_threshold', type=float, default=None, help=f'Minimum p99(x_seq) rain rate (mm/h) for a sequence to count as active rain rather than dry/clear-sky (default: Config.active_rain_threshold_mmh = {Config.active_rain_threshold_mmh}). Example: --active_rain_threshold 2.0')
    parser.add_argument('--fresh_subset', action='store_true', help='Ignore any existing subset_manifest_csv and sample a new active-rain subset from scratch (overwriting the manifest file). Use this after deliberately changing --subset_size or --active_rain_threshold.')
    parser.add_argument('--legacy_selection', action='store_true', help='Disable the active-rain subset entirely and fall back to the old select_top_p99/num_samples/select_on_the_hour behavior (Config.use_active_rain_subset=False).')
    parser.add_argument('--num_processes', type=int, default=None, help=f'How many samples to process simultaneously, each in its own worker process (default: Config.num_processes = {Config.num_processes}, i.e. cpu_count()-1 on this machine). Set to 1 to run sequentially. Example: --num_processes 8')
    parser.add_argument('--quick_test', type=int, default=None, help='Sanity-check mode: after normal sample selection (subset or legacy), keep only the first N selected samples, ignoring the resume cache. Use e.g. --quick_test 10 for a fast 10-sample check before committing to a full run. Does not touch subset_manifest_csv.')
    parser.add_argument('--quick_test_outdir', type=str, default=None, help='Optional separate output dir for --quick_test runs, so quick-test CSVs never mix with / get "resumed" into your full-run CSVs. Defaults to <out_dir>_quicktest when --quick_test is set and this is not given.')
    parser.add_argument('--motion_method', type=str, default=None, choices=['LK', 'VET', 'proesmans'], help=f"Optical-flow motion estimator (default: Config.motion_method = {Config.motion_method!r}). 'LK' is sparse corner-tracking + interpolation (original script); 'VET' and 'proesmans' are dense variational methods that tend to be more robust on precip fields with large uniform zero-background regions. Try --motion_method VET if LK is giving noisy motion diagnostics or weak CSI/ETS at short lead time.")
    parser.add_argument('--extrap_interp_order', type=int, default=None, choices=[0, 1, 3], help=f'Interpolation order for semi-Lagrangian advection (default: Config.extrap_interp_order = {Config.extrap_interp_order}). 0=nearest-neighbor (preserves peak intensity, but can spatially smear/enlarge peaks under a spatially-varying velocity field -- blocky artifacts). 1=bilinear (a smoother middle ground). 3=cubic (pysteps default before our change; smooths peaks the most). Compare against the default if predicted extreme-value regions look larger/blockier than ground truth.')
    parser.add_argument('--save_motion_diagnostics', action='store_true', help='Save a velocity-field quiver-plot PNG for every sample processed, so you can visually sanity-check the motion field (smooth and physically plausible vs noisy/near-zero). Recommended alongside --quick_test when comparing motion methods.')
    parser.add_argument('--save_overlay_diagnostics', action='store_true', help='Save a PNG per sample/model/horizon (t+15, t+30 by default) with the ground-truth rain boundary drawn as a black contour on top of the predicted field. Makes exact displacement error directly visible -- use this to check whether "looks similar" cases are actually pixel-aligned or just visually close.')
    parser.add_argument('--nowcast_method', type=str, default=None, choices=['extrapolation', 'sprog'], help=f"Forecast method (default: Config.nowcast_method = {Config.nowcast_method!r}). 'extrapolation' is pure translation (no intensity evolution). 'sprog' adds an AR(2) cascade decay model on top of the same velocity field -- usually a stronger classical baseline for CSI/ETS at higher thresholds. Requires ar_order+1 (default 3) precip frames, which lk_num_frames already provides.")
    parser.add_argument('--sprog_ar_order', type=int, default=None, help=f'AR order for sprog (default: Config.sprog_ar_order = {Config.sprog_ar_order}). Needs ar_order+1 precip frames -- set Config.lk_num_frames >= ar_order+1. Try ar_order=4 (uses the full 5-frame history) if the default ar_order=2 (3 frames) gives an under-conditioned, over-damped AR fit.')
    parser.add_argument('--lk_num_frames', type=int, default=None, help=f'History frames used for motion estimation (default: Config.lk_num_frames = {Config.lk_num_frames}). VET/proesmans are internally capped at the last 3 regardless of this value.')
    parser.add_argument('--persistence_blend_max_horizon', type=int, default=None, help=f'Lead time (min) at which the persistence blend weight reaches 0 (default: Config.persistence_blend_max_horizon = {Config.persistence_blend_max_horizon}).')
    parser.add_argument('--disable_persistence_blend', action='store_true', help='Turn off the persistence blend entirely (Config.use_persistence_blend = False), i.e. pure motion-based forecast with no t+0 blending.')
    parser.add_argument('--out_dir', type=str, default=None, help='Override Config.out_dir entirely (takes priority over --quick_test_outdir). Use this for tuning sweeps so each trial writes to its own folder.')
    parser.add_argument('--single_datetime', type=str, default=None, help="Evaluate exactly ONE sample by its exact 'YYYY-MM-DD HH:MM:SS' datetime string (must match a row in metadata_csv), bypassing all subset/legacy selection. Automatically saves its comparison figure. Example: --single_datetime '2024-06-30 01:00:00'")
    parser.add_argument('--save_image_for', type=str, default=None, help="Run the NORMAL subset/legacy selection (e.g. --subset_size 300) but only save a comparison figure for the sample(s) matching this exact 'YYYY-MM-DD HH:MM:SS' datetime (comma-separate for more than one). Unlike --single_datetime, this still evaluates the full subset for metrics -- it just limits which figures get saved. Example: --save_image_for '2024-06-30 01:00:00'")
    parser.add_argument('--subset_manifest_csv', type=str, default=None, help='Override Config.subset_manifest_csv. Point every trial in a tuning sweep at the SAME manifest so all configs are compared on identical samples.')
    args = parser.parse_args()
    cfg = Config()
    if args.num_samples is not None:
        cfg.num_samples = None if args.num_samples == 0 else args.num_samples
    if args.subset_size is not None:
        cfg.subset_size = args.subset_size
    if args.active_rain_threshold is not None:
        cfg.active_rain_threshold_mmh = args.active_rain_threshold
    if args.fresh_subset:
        cfg.use_existing_subset_manifest = False
    if args.legacy_selection:
        cfg.use_active_rain_subset = False
    if args.num_processes is not None:
        cfg.num_processes = args.num_processes
    if args.motion_method is not None:
        cfg.motion_method = args.motion_method
    if args.extrap_interp_order is not None:
        cfg.extrap_interp_order = args.extrap_interp_order
    if args.nowcast_method is not None:
        cfg.nowcast_method = args.nowcast_method
    if args.sprog_ar_order is not None:
        cfg.sprog_ar_order = args.sprog_ar_order
    if args.lk_num_frames is not None:
        cfg.lk_num_frames = args.lk_num_frames
    if args.persistence_blend_max_horizon is not None:
        cfg.persistence_blend_max_horizon = args.persistence_blend_max_horizon
    if args.disable_persistence_blend:
        cfg.use_persistence_blend = False
    if args.subset_manifest_csv is not None:
        cfg.subset_manifest_csv = args.subset_manifest_csv
    if args.save_image_for is not None:
        cfg.save_images = True
        cfg.save_images_only_datetimes = [s.strip() for s in args.save_image_for.split(',')]
    if args.save_motion_diagnostics:
        cfg.save_motion_diagnostics = True
    if args.save_overlay_diagnostics:
        cfg.save_overlay_diagnostics = True

    if args.quick_test is not None:
        cfg.out_dir = args.quick_test_outdir if args.quick_test_outdir else (cfg.out_dir + '_quicktest')
        cfg.detailed_metrics_csv = os.path.join(cfg.out_dir, 'lk_extrapolation_radar_multimodal_metrics.csv')
        cfg.mean_metrics_csv = os.path.join(cfg.out_dir, 'lk_extrapolation_radar_multimodal_mean_metrics.csv')
        if os.path.exists(cfg.detailed_metrics_csv):
            os.remove(cfg.detailed_metrics_csv)

    if args.out_dir is not None:
        cfg.out_dir = args.out_dir
        cfg.detailed_metrics_csv = os.path.join(cfg.out_dir, 'lk_extrapolation_radar_multimodal_metrics.csv')
        cfg.mean_metrics_csv = os.path.join(cfg.out_dir, 'lk_extrapolation_radar_multimodal_mean_metrics.csv')
        if os.path.exists(cfg.detailed_metrics_csv):
            os.remove(cfg.detailed_metrics_csv)

    os.makedirs(cfg.out_dir, exist_ok=True)
    print()
    print('=' * 80)
    print('pySTEPS RADAR + MULTIMODAL LK EXTRAPOLATION COMPARISON')
    print('=' * 80)
    print(f'Radar     : {cfg.radar_base}')
    print(f'Satellite : {cfg.sat_base}')
    print(f'Channels  : {cfg.channels}')
    print(f'Horizons  : {cfg.horizons}')
    print(f'Forecast   : {cfg.motion_method} motion + {cfg.nowcast_method} nowcast')
    print(f'History frames: {cfg.lk_num_frames} (spanning {(cfg.lk_num_frames - 1) * cfg.stride_minutes} min back)')
    print(f'extrap interp_order: {cfg.extrap_interp_order}  |  persistence_blend: {cfg.use_persistence_blend} (<= {cfg.persistence_blend_max_horizon} min)')
    if args.quick_test is not None:
        print(f'QUICK TEST MODE: first {args.quick_test} samples only -> {cfg.out_dir}')
    print('=' * 80)

    channel_stats = load_channel_stats(cfg)
    if not os.path.exists(cfg.metadata_csv):
        raise FileNotFoundError(f'Metadata not found:\n{cfg.metadata_csv}')
    df = pd.read_csv(cfg.metadata_csv)

    if args.single_datetime is not None:
        selected = df[df['datetime'] == args.single_datetime].reset_index(drop=True)
        if selected.empty:
            raise ValueError(
                f"--single_datetime '{args.single_datetime}' not found in {cfg.metadata_csv}. "
                f"Must match an existing 'datetime' value exactly (format: 'YYYY-MM-DD HH:MM:SS')."
            )
        cfg.save_images = True
        print(f"\n[single_datetime mode] Evaluating exactly one sample: {args.single_datetime}")
        print(f"[single_datetime mode] Figure will be saved to: {cfg.out_dir}\n")

    elif cfg.use_active_rain_subset:
        if 'p99(x_seq)' not in df.columns:
            raise ValueError("use_active_rain_subset=True but the metadata CSV has no 'p99(x_seq)' column to filter dry frames by. Set Config.use_active_rain_subset = False to fall back to the legacy selection, or point active_rain_threshold_mmh at whatever rain-activity column your CSV actually has.")
        active_df = df[df['p99(x_seq)'] >= cfg.active_rain_threshold_mmh].reset_index(drop=True)
        print(f'\nActive-rain filter (p99(x_seq) >= {cfg.active_rain_threshold_mmh:.2f} mm/h): {len(active_df)} / {len(df)} candidates ({len(df) - len(active_df)} dry/near-dry sequences excluded)')
        if len(active_df) == 0:
            raise ValueError(f'No sequences in {cfg.metadata_csv} have p99(x_seq) >= {cfg.active_rain_threshold_mmh}. Lower Config.active_rain_threshold_mmh.')
        manifest_path = cfg.subset_manifest_csv
        manifest_exists = os.path.exists(manifest_path)
        if cfg.use_existing_subset_manifest and manifest_exists:
            manifest_df = pd.read_csv(manifest_path)
            selected = active_df[active_df['datetime'].isin(manifest_df['datetime'])].reset_index(drop=True)
            print(f'Loaded existing subset manifest: {manifest_path}\n  ({len(selected)} / {len(manifest_df)} manifest entries still present in the active-rain pool)')
        else:
            n = min(cfg.subset_size, len(active_df))
            if n < cfg.subset_size:
                print(f'  [WARN] Requested subset_size={cfg.subset_size} but only {n} active-rain sequences are available -- using all of them.')
            selected = active_df.sample(n=n, random_state=cfg.subset_seed).reset_index(drop=True)
            os.makedirs(os.path.dirname(manifest_path) or '.', exist_ok=True)
            selected[['datetime']].to_csv(manifest_path, index=False)
            print(f"Sampled a fresh subset ({len(selected)} sequences, seed={cfg.subset_seed}) and wrote manifest to:\n  {manifest_path}\n  Point every other evaluation script's subset_manifest_csv at this same file to guarantee they all score the identical sequences.")
        if cfg.select_on_the_hour:
            parsed_minute = pd.to_datetime(selected['datetime'], format='%Y-%m-%d %H:%M:%S').dt.minute
            before = len(selected)
            selected = selected[parsed_minute == 45].reset_index(drop=True)
            print(f'  [WARN] select_on_the_hour=True further narrowed the active-rain subset from {before} to {len(selected)} sequences (only :45-past-the-hour events kept). This is a figure-timestamp cosmetic constraint -- consider turning it off for bulk metric runs.')
    else:
        if cfg.select_on_the_hour:
            parsed_minute = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S').dt.minute
            df = df[parsed_minute == 45].reset_index(drop=True)
            if len(df) == 0:
                raise ValueError(f'select_on_the_hour=True but no rows in {cfg.metadata_csv} have a datetime with minute == 45. Set Config.select_on_the_hour = False to use all candidate times instead.')
            print(f'\nRestricted to events at :45 past the hour: {len(df)} candidates')
        if cfg.select_top_p99 and 'p99(x_seq)' in df.columns:
            if cfg.num_samples is None:
                selected = df.sort_values('p99(x_seq)', ascending=False).reset_index(drop=True)
            else:
                selected = df.nlargest(cfg.num_samples, 'p99(x_seq)').reset_index(drop=True)
        elif cfg.num_samples is None:
            selected = df.reset_index(drop=True)
        else:
            selected = df.head(cfg.num_samples).reset_index(drop=True)
        if cfg.select_top_p99:
            print("  (top-P99 selection -- biased toward fast/compact convective cells; see inference_pysteps_baseline.py's sample_selection='random' option if you want a representative sample instead)")

    if args.quick_test is not None:
        selected = selected.head(args.quick_test).reset_index(drop=True)

    print(f'\nSamples selected: {len(selected)}')
    if cfg.save_images and cfg.save_images_only_datetimes is not None:
        print(f'  Figures will be saved for {len(cfg.save_images_only_datetimes)} specific sample(s) only: {cfg.save_images_only_datetimes}')
    elif cfg.save_images and len(selected) > 50:
        print(f'  [WARN] save_images=True with {len(selected)} samples will generate that many PNGs and add real runtime -- consider Config.save_images = False for bulk metric runs, or use --save_image_for to limit to specific samples.')

    already_done = set()
    if args.quick_test is None and args.single_datetime is None and os.path.exists(cfg.detailed_metrics_csv):
        prior_df = pd.read_csv(cfg.detailed_metrics_csv)
        already_done = set(prior_df['datetime'].unique())
        print(f'\nResume: found {len(already_done)} already-completed sample datetimes in {cfg.detailed_metrics_csv}')

    remaining = selected[~selected['datetime'].isin(already_done)].reset_index(drop=True)
    print(f'Remaining to process: {len(remaining)} / {len(selected)}')

    worker_args = [(sample_idx, row, cfg, channel_stats) for sample_idx, row in remaining.iterrows()]

    csv_header_written = os.path.exists(cfg.detailed_metrics_csv)

    def _append_results(rows):
        nonlocal csv_header_written
        if not rows:
            return
        chunk_df = pd.DataFrame(rows)
        chunk_df.to_csv(cfg.detailed_metrics_csv, mode='a', index=False, header=not csv_header_written)
        csv_header_written = True

    num_processes = 1 if (args.quick_test is not None or args.single_datetime is not None) else cfg.num_processes

    if worker_args:
        if num_processes > 1 and len(worker_args) > 1:
            print(f'\nRunning {len(worker_args)} samples across {num_processes} worker processes (each sample uses deterministic LK extrapolation)...\n')
            with mp.Pool(processes=num_processes) as pool:
                for result in pool.imap_unordered(process_one_sample, worker_args):
                    _append_results(result)
        else:
            print(f'\nRunning {len(worker_args)} samples sequentially...\n')
            for a in worker_args:
                _append_results(process_one_sample(a))
    else:
        print('\nAll selected samples already completed -- nothing to run, computing mean from existing results.')

    if not os.path.exists(cfg.detailed_metrics_csv):
        print('\nNo results generated.')
        return

    metrics_df = pd.read_csv(cfg.detailed_metrics_csv)
    mean_df = compute_pooled_mean(metrics_df, cfg)
    mean_df.to_csv(cfg.mean_metrics_csv, index=False)

    print()
    print('=' * 80)
    print('FINAL POOLED MEAN RESULTS  (CSI/ETS pooled from summed counts, not averaged per-sample)')
    print('=' * 80)
    print(mean_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    print()
    print('-' * 80)
    print('QUICK LOOK: Radar-only vs Multimodal at t+15 / t+30 (the horizons you care about)')
    print('-' * 80)
    focus = mean_df[mean_df['horizon_min'].isin(cfg.metric_horizons)].sort_values(['horizon_min', 'model'])
    if not focus.empty:
        display_cols = ['model', 'horizon_min', 'MAE', 'MSE', 'PSNR', 'CSI', 'ETS']
        print(focus[display_cols].to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    print()
    print(f'Detailed CSV:\n{cfg.detailed_metrics_csv}')
    print(f'\nMean CSV (pooled):\n{cfg.mean_metrics_csv}')
    print('\nDone.')


if __name__ == '__main__':
    main()