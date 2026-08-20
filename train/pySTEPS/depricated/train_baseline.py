import os
import json
from datetime import datetime, timedelta

import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage.metrics import structural_similarity

from pysteps import motion, nowcasts

from thresholds import VIZ_OPERATIONAL_MMH, VIZ_EXTREME_MMH
import colors as PALETTE


CLIP_MAX_MMH = 128.0

RAIN_LEVELS = PALETTE.RAIN_LEVELS
RAIN_COLORS = PALETTE.RAIN_COLORS
RAIN_CMAP = PALETTE.RAIN_CMAP
RAIN_NORM = PALETTE.RAIN_NORM
BG_COLOR = PALETTE.BG_COLOR


class Config:
    radar_base = "/home/fe/sajib/scratch/weather-data/radar_de"
    sat_base = "/home/fe/sajib/scratch/weather-data/satellite_de_regridded"
    channels = ["CH7", "CH9"]

    metadata_csv = "metadata_patch/test_fullimage_summer.csv"
    channel_stats_json = "metadata/channel_stats.json"

    out_dir = "PYSTEPS_RADAR_MULTIMODAL"
    detailed_metrics_csv = os.path.join(out_dir, "pysteps_radar_multimodal_metrics.csv")
    mean_metrics_csv = os.path.join(out_dir, "pysteps_radar_multimodal_mean_metrics.csv")

    num_in_frames = 5
    stride_minutes = 5
    horizons = [15, 30, 45, 60]
    metric_horizons = [15, 30, 45, 60]

    ar_order = 2
    sprog_num_frames = ar_order + 1
    n_cascade_levels = 8
    rain_threshold = 0.1
    precip_thr_db = 10.0 * np.log10(rain_threshold)
    zerovalue_db = -15.0
    extrap_method = "semilagrangian"
    decomp_method = "fft"
    bandpass_filter_method = "gaussian"
    probmatching_method = None
    conditional = False
    num_workers = 1

    num_samples = None 
    select_top_p99 = True
    select_on_the_hour = True

    specific_datetime = None

    categorical_thresholds = [5.0, 15.0]
    csi_threshold_mmh = 15.0
    psnr_data_range = CLIP_MAX_MMH

    save_images = True
    operational_thr = VIZ_OPERATIONAL_MMH
    extreme_thr = VIZ_EXTREME_MMH


def radar_path(base, dt):
    return os.path.join(base, dt.strftime("%y%m"), dt.strftime("%d"), dt.strftime("%y%m%d_%H%M") + ".npy")


def satellite_path(base, dt, channel):
    return os.path.join(base, dt.strftime("%Y"), dt.strftime("%m"), dt.strftime("%d"), dt.strftime("%H%M%S") + f"_{channel}.npy")


def get_history_times(current_time, num_frames, stride_minutes):
    start = current_time - (num_frames - 1) * timedelta(minutes=stride_minutes)
    return [start + timedelta(minutes=stride_minutes * i) for i in range(num_frames)]


def get_target_times(current_time, horizons):
    return [current_time + timedelta(minutes=h) for h in horizons]


def load_npy(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"\nFile not found:\n{path}")
    return np.load(path).astype(np.float32)


def load_channel_stats(cfg):
    if not os.path.exists(cfg.channel_stats_json):
        print("\nWARNING: channel_stats.json not found.")
        print("Satellite normalization will use per-sequence robust normalization.")
        return None
    with open(cfg.channel_stats_json, "r") as f:
        stats = json.load(f)
    return {int(k): v for k, v in stats.items()}


def load_sample(row, cfg):
    dt = datetime.strptime(row["datetime"], "%Y-%m-%d %H:%M:%S")
    history_times = get_history_times(dt, cfg.sprog_num_frames, cfg.stride_minutes)
    target_times = get_target_times(dt, cfg.metric_horizons)

    radar_history = []
    radar_history_masks = []
    for t in history_times:
        arr = load_npy(radar_path(cfg.radar_base, t))
        valid = np.isfinite(arr)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.clip(arr, 0.0, CLIP_MAX_MMH)
        radar_history.append(arr)
        radar_history_masks.append(valid)
    radar_history = np.stack(radar_history, axis=0).astype(np.float32)
    radar_history_masks = np.stack(radar_history_masks, axis=0)

    satellite_history = {}
    for channel in cfg.channels:
        channel_frames = []
        for t in history_times:
            arr = load_npy(satellite_path(cfg.sat_base, t, channel))
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            channel_frames.append(arr)
        satellite_history[channel] = np.stack(channel_frames, axis=0).astype(np.float32)

    gt = {}
    masks = {}
    for h, t in zip(cfg.metric_horizons, target_times):
        arr = load_npy(radar_path(cfg.radar_base, t))
        valid = np.isfinite(arr)
        clean = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        clean = np.clip(clean, 0.0, CLIP_MAX_MMH)
        gt[h] = clean.astype(np.float32)
        masks[h] = valid

    return radar_history, radar_history_masks, satellite_history, gt, masks, dt, history_times, target_times


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
    ch_number = int(channel.replace("CH", ""))

    if channel_stats is not None and ch_number in channel_stats:
        mean = float(channel_stats[ch_number]["mean"])
        std = float(channel_stats[ch_number]["std"])
        seq = (seq - mean) / (std + 1e-8)
    else:
        finite = np.isfinite(seq)
        if not np.any(finite):
            return np.zeros_like(seq)
        median = np.median(seq[finite])
        p25 = np.percentile(seq[finite], 25)
        p75 = np.percentile(seq[finite], 75)
        scale = max(p75 - p25, 1e-6)
        seq = (seq - median) / scale

    seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
    low = np.percentile(seq, 1)
    high = np.percentile(seq, 99)
    if high - low < 1e-8:
        return np.zeros_like(seq)
    seq = (seq - low) / (high - low)
    seq = np.clip(seq, 0.0, 1.0)
    return seq.astype(np.float32)


def estimate_lk_motion(frames, name):
    frames = np.asarray(frames, dtype=np.float32)
    frames = np.nan_to_num(frames, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"       LK motion: {name}")
    lk = motion.get_method("LK")
    velocity = lk(frames)
    velocity = np.nan_to_num(velocity, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    mean_speed = np.mean(np.sqrt(velocity[0] ** 2 + velocity[1] ** 2))
    print(f"          mean speed = {mean_speed:.4f} pixels/timestep")
    return velocity


def estimate_radar_motion(radar_db):
    return estimate_lk_motion(radar_db, "Radar")


def estimate_multimodal_motion(radar_db, satellite_history, channel_stats, cfg):
    v_radar = estimate_lk_motion(radar_db, "Radar")
    velocities = [v_radar]

    for channel in cfg.channels:
        sat_seq = satellite_history[channel][-cfg.sprog_num_frames:]
        sat_normalized = normalize_satellite_sequence(sat_seq, channel, channel_stats)
        v_sat = estimate_lk_motion(sat_normalized, channel)
        velocities.append(v_sat)

    velocity_stack = np.stack(velocities, axis=0)
    multimodal_velocity = np.median(velocity_stack, axis=0)
    multimodal_velocity = np.nan_to_num(multimodal_velocity, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    mean_speed = np.mean(np.sqrt(multimodal_velocity[0] ** 2 + multimodal_velocity[1] ** 2))
    print(f"       Combined multimodal mean speed = {mean_speed:.4f}")

    return multimodal_velocity


def run_sprog_with_velocity(radar_history, velocity, cfg):
    required_frames = cfg.ar_order + 1
    radar_input = radar_history[-required_frames:]
    radar_db = rainrate_to_db(radar_input, threshold=cfg.rain_threshold, zerovalue=cfg.zerovalue_db)
    radar_db = np.nan_to_num(radar_db, nan=cfg.zerovalue_db, posinf=cfg.zerovalue_db, neginf=cfg.zerovalue_db)

    n_steps = max(cfg.metric_horizons) // cfg.stride_minutes
    sprog = nowcasts.get_method("sprog")

    forecast_db = sprog(
        radar_db, velocity, n_steps,
        precip_thr=cfg.precip_thr_db,
        n_cascade_levels=cfg.n_cascade_levels,
        extrap_method=cfg.extrap_method,
        decomp_method=cfg.decomp_method,
        bandpass_filter_method=cfg.bandpass_filter_method,
        ar_order=cfg.ar_order,
        conditional=cfg.conditional,
        probmatching_method=cfg.probmatching_method,
        num_workers=cfg.num_workers,
        domain="spatial",
    )

    forecast_db = np.asarray(forecast_db, dtype=np.float32)
    forecast_mm = db_to_rainrate(forecast_db, cfg)

    predictions = {}
    for horizon in cfg.metric_horizons:
        step = horizon // cfg.stride_minutes
        index = step - 1
        predictions[horizon] = np.clip(forecast_mm[index], 0.0, CLIP_MAX_MMH).astype(np.float32)

    return predictions


def run_both_baselines(radar_history, satellite_history, channel_stats, cfg):
    print("\n    Preparing S-PROG input...")
    radar_recent = radar_history[-cfg.sprog_num_frames:]
    radar_db = rainrate_to_db(radar_recent, threshold=cfg.rain_threshold, zerovalue=cfg.zerovalue_db)

    print("\n    [1/2] pySTEPS RADAR")
    radar_velocity = estimate_radar_motion(radar_db)
    radar_predictions = run_sprog_with_velocity(radar_history, radar_velocity, cfg)

    print("\n    [2/2] pySTEPS MULTIMODAL (Radar + CH7 + CH9 motion)")
    multimodal_velocity = estimate_multimodal_motion(radar_db, satellite_history, channel_stats, cfg)
    multimodal_predictions = run_sprog_with_velocity(radar_history, multimodal_velocity, cfg)

    return radar_predictions, multimodal_predictions, radar_velocity, multimodal_velocity


def contingency_counts(pred, target, threshold):
    pred_event = pred >= threshold
    target_event = target >= threshold
    hits = np.sum(pred_event & target_event)
    misses = np.sum((~pred_event) & target_event)
    false_alarms = np.sum(pred_event & (~target_event))
    correct_negatives = np.sum((~pred_event) & (~target_event))
    return float(hits), float(misses), float(false_alarms), float(correct_negatives)


def compute_csi(pred, target, threshold):
    hits, misses, false_alarms, _ = contingency_counts(pred, target, threshold)
    denominator = hits + misses + false_alarms
    if denominator == 0:
        return np.nan
    return hits / denominator


def compute_ets(pred, target, threshold):
    hits, misses, false_alarms, correct_negatives = contingency_counts(pred, target, threshold)
    total = hits + misses + false_alarms + correct_negatives
    if total == 0:
        return np.nan
    random_hits = (hits + misses) * (hits + false_alarms) / total
    denominator = hits + misses + false_alarms - random_hits
    if denominator == 0:
        return np.nan
    return (hits - random_hits) / denominator


def compute_psnr(pred, target, data_range):
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10((data_range ** 2) / mse)


def compute_metrics(prediction, target, mask, cfg):
    valid = mask.astype(bool) & np.isfinite(prediction) & np.isfinite(target)
    if not np.any(valid):
        return None

    active = valid & ((target >= cfg.rain_threshold) | (prediction >= cfg.rain_threshold))
    MIN_ACTIVE_PIXELS = 100

    if active.sum() < MIN_ACTIVE_PIXELS:
        mae = mse = rmse = psnr = ssim = float("nan")
    else:
        p = prediction[active]
        y = target[active]
        mae = np.mean(np.abs(p - y))
        mse = np.mean((p - y) ** 2)
        rmse = np.sqrt(mse)
        psnr = compute_psnr(p, y, cfg.psnr_data_range)

        pred_db = rainrate_to_db(np.where(valid, prediction, 0.0), threshold=cfg.rain_threshold, zerovalue=cfg.zerovalue_db)
        target_db = rainrate_to_db(np.where(valid, target, 0.0), threshold=cfg.rain_threshold, zerovalue=cfg.zerovalue_db)
        db_range = 10.0 * np.log10(cfg.psnr_data_range) - cfg.zerovalue_db

        _, ssim_map = structural_similarity(
            target_db, pred_db, data_range=db_range,
            gaussian_weights=True, sigma=1.5, use_sample_covariance=False, full=True,
        )
        ssim = float(ssim_map[active].mean())

    result = {"MAE": float(mae), "MSE": float(mse), "RMSE": float(rmse), "PSNR": float(psnr), "SSIM": float(ssim)}
    result["CSI"] = compute_csi(prediction[valid], target[valid], cfg.csi_threshold_mmh)

    for threshold in cfg.categorical_thresholds:
        suffix = str(int(threshold))
        result[f"ETS@{suffix}"] = compute_ets(prediction[valid], target[valid], threshold)

    return result


def detect_storms_two_level(rain_mm_h, operational_thr, extreme_thr, min_pixels=200, disk_size=4,
                             min_area_km2=200.0, max_area_km2=10000.0, pixel_area_km2=1.0):
    from skimage.morphology import closing, remove_small_objects, disk as sk_disk
    from scipy.ndimage import label as sp_label

    valid = np.isfinite(rain_mm_h)
    if valid.sum() == 0:
        empty = np.zeros_like(rain_mm_h, dtype=bool)
        return empty, empty, 0, 0

    mask_op = valid & (rain_mm_h >= operational_thr)
    if mask_op.sum() == 0:
        empty = np.zeros_like(rain_mm_h, dtype=bool)
        return empty, empty, 0, 0

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
        return final_op, mask_ext, n_op, 0

    mask_ext = closing(mask_ext, sk_disk(2))
    mask_ext = remove_small_objects(mask_ext, min_size=50)
    _, n_ext = sp_label(mask_ext)
    return final_op, mask_ext, n_op, n_ext


def render_precip(ax, image, valid_mask=None, operational_thr=None, extreme_thr=None):
    ax.set_facecolor(BG_COLOR)

    display_image = np.ma.masked_less(image, 0.1)
    if valid_mask is not None:
        display_image = np.ma.masked_where(~valid_mask, display_image)

    im = ax.imshow(display_image, cmap=RAIN_CMAP, norm=RAIN_NORM, interpolation="nearest")

    if operational_thr is not None and extreme_thr is not None:
        field = np.where(valid_mask, image, np.nan) if valid_mask is not None else image
        op, ext, n_op, n_ext = detect_storms_two_level(field, operational_thr, extreme_thr)
        if n_op > 0:
            ov = np.zeros((*op.shape, 4))
            ov[op] = PALETTE.OP_OVERLAY_RGBA
            ax.imshow(ov, interpolation="nearest")
        if n_ext > 0:
            ov = np.zeros((*ext.shape, 4))
            ov[ext] = PALETTE.EXT_OVERLAY_RGBA
            ax.imshow(ov, interpolation="nearest")

    ax.axis("off")
    return im


def save_comparison_figure(gt, radar_preds, multimodal_preds, masks, dt, cfg, sample_index,
                            t0_frame=None, t0_mask=None, horizons_to_show=None, filename_suffix=""):
    RADAR_COLOR = PALETTE.RADAR_COLOR
    MULTIMODAL_COLOR = PALETTE.MULTIMODAL_COLOR
    GT_COLOR = PALETTE.GT_COLOR

    fig = plt.figure(figsize=(20, 12), facecolor="white")

    show_t0 = t0_frame is not None
    base_horizons = horizons_to_show if horizons_to_show is not None else cfg.horizons
    display_horizons = ([0] if show_t0 else []) + list(base_horizons)
    n_cols = len(display_horizons)

    gs = fig.add_gridspec(3, n_cols + 1, width_ratios=[1] * n_cols + [0.055],
                           left=0.10, right=0.94, bottom=0.08, top=0.86, hspace=0.20, wspace=0.05)

    last_im = None

    for col, h in enumerate(display_horizons):
        ax = fig.add_subplot(gs[0, col])
        panel_img, panel_mask = (t0_frame, t0_mask) if h == 0 else (gt[h], masks[h])
        last_im = render_precip(ax, panel_img, valid_mask=panel_mask, operational_thr=cfg.operational_thr, extreme_thr=cfg.extreme_thr)
        target_time = dt + timedelta(minutes=h)
        title_text = f"Ground Truth  |  Analysis (t+0)\n{target_time.strftime('%Y-%m-%d %H:%M')}" if h == 0 else f"Ground Truth  |  GT t+{h}min\n{target_time.strftime('%Y-%m-%d %H:%M')}"
        ax.set_title(title_text, fontsize=9, fontweight="bold", pad=10)

    for col, h in enumerate(display_horizons):
        ax = fig.add_subplot(gs[1, col])
        panel_img, panel_mask = (t0_frame, t0_mask) if h == 0 else (radar_preds[h], masks[h])
        render_precip(ax, panel_img, valid_mask=panel_mask, operational_thr=cfg.operational_thr, extreme_thr=cfg.extreme_thr)
        target_time = dt + timedelta(minutes=h)
        title_text = f"PySteps Radar  |  Analysis (t+0)\n{target_time.strftime('%Y-%m-%d %H:%M')}" if h == 0 else f"PySteps Radar  |  Pred t+{h}min\n{target_time.strftime('%Y-%m-%d %H:%M')}"
        ax.set_title(title_text, fontsize=9, fontweight="bold", color=RADAR_COLOR, pad=10)

    for col, h in enumerate(display_horizons):
        ax = fig.add_subplot(gs[2, col])
        panel_img, panel_mask = (t0_frame, t0_mask) if h == 0 else (multimodal_preds[h], masks[h])
        render_precip(ax, panel_img, valid_mask=panel_mask, operational_thr=cfg.operational_thr, extreme_thr=cfg.extreme_thr)
        target_time = dt + timedelta(minutes=h)
        title_text = f"PySteps Multimodal  |  Analysis (t+0)\n{target_time.strftime('%Y-%m-%d %H:%M')}" if h == 0 else f"PySteps Multimodal  |  Pred t+{h}min\n{target_time.strftime('%Y-%m-%d %H:%M')}"
        ax.set_title(title_text, fontsize=9, fontweight="bold", color=MULTIMODAL_COLOR, pad=10)

    fig.text(0.055, 0.76, "Ground\nTruth", ha="center", va="center", fontsize=14, fontweight="bold", color=GT_COLOR)
    fig.text(0.055, 0.50, "PySteps\nRadar", ha="center", va="center", fontsize=14, fontweight="bold", color=RADAR_COLOR)
    fig.text(0.055, 0.245, "PySteps\nMultimodal", ha="center", va="center", fontsize=14, fontweight="bold", color=MULTIMODAL_COLOR)

    cbar_ax = fig.add_subplot(gs[:, n_cols])
    cbar = fig.colorbar(last_im, cax=cbar_ax, ticks=RAIN_LEVELS, spacing="proportional")
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("mm / h", fontsize=10, fontweight="bold")

    cbar_ax.axhline(y=cfg.operational_thr, linestyle="--", linewidth=1.3, color="#f0a000")
    cbar_ax.annotate(f"OPR\n{cfg.operational_thr:.0f}", xy=(1.15, cfg.operational_thr), xycoords=("axes fraction", "data"),
                      fontsize=7, color="#d99200", fontweight="bold", va="center",
                      bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#d99200", lw=0.8))

    cbar_ax.axhline(y=cfg.extreme_thr, linestyle="--", linewidth=1.3, color="#dd5500")
    cbar_ax.annotate(f"EXT\n{cfg.extreme_thr:.0f}", xy=(1.15, cfg.extreme_thr), xycoords=("axes fraction", "data"),
                      fontsize=7, color="#dd5500", fontweight="bold", va="center",
                      bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#dd5500", lw=0.8))

    fig.suptitle("Comparison of Ground Truth and pySTEPS Baseline Predictions", fontsize=17, fontweight="bold", y=0.98)

    handles = [
        mpatches.Patch(color=GT_COLOR, label="Ground Truth"),
        mpatches.Patch(color=RADAR_COLOR, label="PySteps Radar"),
        mpatches.Patch(color=MULTIMODAL_COLOR, label="PySteps Multimodal (Radar+CH7+CH9)"),
        mpatches.Patch(color=PALETTE.OP_LEGEND_RGBA, label=f"Operational >={cfg.operational_thr:.0f} mm/h"),
        mpatches.Patch(color=PALETTE.EXT_LEGEND_RGBA, label=f"Extreme >={cfg.extreme_thr:.0f} mm/h"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, frameon=True, bbox_to_anchor=(0.52, 0.025))

    output_path = os.path.join(cfg.out_dir, f"pysteps_sprog_comparison_sample{sample_index:03d}_{dt.strftime('%Y-%m-%d')}{filename_suffix}.png")
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n    Figure saved:\n    {output_path}")


def main():
    parser = argparse.ArgumentParser(description="pySTEPS Radar + Multimodal S-PROG baseline")
    parser.add_argument("--n_cascade_levels", type=int, default=None,
                         help=f"Number of S-PROG cascade levels (default: Config.n_cascade_levels = {Config.n_cascade_levels}).")
    args = parser.parse_args()

    cfg = Config()
    if args.n_cascade_levels is not None:
        cfg.n_cascade_levels = args.n_cascade_levels

    os.makedirs(cfg.out_dir, exist_ok=True)

    print()
    print("=" * 80)
    print("pySTEPS RADAR + MULTIMODAL S-PROG COMPARISON")
    print("=" * 80)
    print(f"Radar     : {cfg.radar_base}")
    print(f"Satellite : {cfg.sat_base}")
    print(f"Channels  : {cfg.channels}")
    print(f"Horizons  : {cfg.horizons}")
    print(f"S-PROG AR : {cfg.ar_order}")
    print(f"Cascade levels: {cfg.n_cascade_levels}" + ("  (from --n_cascade_levels)" if args.n_cascade_levels is not None else "  (Config default)"))
    print(f"History frames: {cfg.sprog_num_frames} (spanning {(cfg.sprog_num_frames - 1) * cfg.stride_minutes} min back)")
    print("=" * 80)

    channel_stats = load_channel_stats(cfg)

    if not os.path.exists(cfg.metadata_csv):
        raise FileNotFoundError(f"Metadata not found:\n{cfg.metadata_csv}")
    df = pd.read_csv(cfg.metadata_csv)

    if cfg.specific_datetime is not None:
        selected = df[df["datetime"] == cfg.specific_datetime].reset_index(drop=True)
        if len(selected) == 0:
            raise ValueError(f"specific_datetime={cfg.specific_datetime!r} not found in {cfg.metadata_csv}.")
        print(f"\nUsing specific_datetime override: {cfg.specific_datetime} ({len(selected)} row)")

    else:
        if cfg.select_on_the_hour:
            parsed_minute = pd.to_datetime(df["datetime"], format="%Y-%m-%d %H:%M:%S").dt.minute
            df = df[parsed_minute == 45].reset_index(drop=True)
            if len(df) == 0:
                raise ValueError(f"select_on_the_hour=True but no rows in {cfg.metadata_csv} have a datetime with minute == 45.")
            print(f"\nRestricted to on-the-hour events (:00): {len(df)} candidates")

        if cfg.select_top_p99 and "p99(x_seq)" in df.columns:
            selected = df.sort_values("p99(x_seq)", ascending=False).reset_index(drop=True) if cfg.num_samples is None else df.nlargest(cfg.num_samples, "p99(x_seq)").reset_index(drop=True)
        else:
            selected = df.reset_index(drop=True) if cfg.num_samples is None else df.head(cfg.num_samples).reset_index(drop=True)

    print(f"\nSamples selected: {len(selected)}")
    if cfg.specific_datetime is None and cfg.select_top_p99:
        print("  (top-P99 selection)")

    all_metrics = []

    for sample_idx, row in selected.iterrows():
        print()
        print("=" * 80)
        print(f"SAMPLE {sample_idx + 1}/{len(selected)}")
        print(f"Datetime: {row['datetime']}")
        print("=" * 80)

        try:
            radar_history, radar_history_masks, satellite_history, gt, masks, dt, history_times, target_times = load_sample(row, cfg)
        except Exception as error:
            print(f"\nLOAD ERROR:\n{error}")
            continue

        print("\n    Historical observations:")
        for t in history_times:
            print(f"       {t.strftime('%Y-%m-%d %H:%M')}")

        print("\n    S-PROG input:")
        for t in history_times[-cfg.sprog_num_frames:]:
            print(f"       {t.strftime('%Y-%m-%d %H:%M')}")

        print("\n    Targets:")
        for horizon, t in zip(cfg.metric_horizons, target_times):
            print(f"       +{horizon:02d}: {t.strftime('%Y-%m-%d %H:%M')}")

        print(f"\n    Radar shape : {radar_history.shape}")
        for ch in cfg.channels:
            print(f"    {ch} shape   : {satellite_history[ch].shape}")

        radar_hw = radar_history.shape[-2:]
        for ch in cfg.channels:
            sat_hw = satellite_history[ch].shape[-2:]
            if sat_hw != radar_hw:
                raise ValueError(f"{ch} grid {sat_hw} does not match radar grid {radar_hw}.\nUse the regridded satellite data.")

        try:
            radar_predictions, multimodal_predictions, radar_velocity, multimodal_velocity = run_both_baselines(radar_history, satellite_history, channel_stats, cfg)
        except Exception as error:
            print(f"\nFORECAST ERROR:\n{error}")
            continue

        print("\n" + "-" * 80)
        print("METRICS")
        print("-" * 80)

        for model_name, predictions in [("PySteps_Radar", radar_predictions), ("PySteps_Multimodal", multimodal_predictions)]:
            print(f"\n    {model_name}")
            for horizon in cfg.metric_horizons:
                metrics = compute_metrics(predictions[horizon], gt[horizon], masks[horizon], cfg)
                if metrics is None:
                    continue
                row_result = {"model": model_name, "datetime": dt.strftime("%Y-%m-%d %H:%M:%S"), "horizon_min": horizon}
                row_result.update(metrics)
                all_metrics.append(row_result)
                print(f"       t+{horizon:02d} | MAE={metrics['MAE']:.4f} | RMSE={metrics['RMSE']:.4f} | SSIM={metrics['SSIM']:.4f} | CSI={metrics['CSI']:.4f} | ETS@5={metrics['ETS@5']:.4f}")

        if cfg.save_images:
            save_comparison_figure(gt, radar_predictions, multimodal_predictions, masks, dt, cfg, sample_idx + 1,
                                    t0_frame=None, t0_mask=None, horizons_to_show=cfg.metric_horizons)

    if not all_metrics:
        print("\nNo results generated.")
        return

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(cfg.detailed_metrics_csv, index=False)

    numeric_cols = [c for c in metrics_df.columns if c not in ["model", "datetime", "horizon_min"]]
    mean_df = metrics_df.groupby(["model", "horizon_min"])[numeric_cols].mean().reset_index()
    mean_df.to_csv(cfg.mean_metrics_csv, index=False)

    print()
    print("=" * 80)
    print("FINAL MEAN RESULTS")
    print("=" * 80)
    print(mean_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print()
    print(f"Detailed CSV:\n{cfg.detailed_metrics_csv}")
    print(f"\nMean CSV:\n{cfg.mean_metrics_csv}")
    print("\nDone.")


if __name__ == "__main__":
    main()