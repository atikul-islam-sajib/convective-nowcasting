import os
import json
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import argparse


def radar_path(base, dt):
    return os.path.join(
        base, dt.strftime('%y%m'), dt.strftime('%d'),
        dt.strftime('%y%m%d_%H%M') + '.npy'
    )


def satellite_path(base, dt, channel):
    return os.path.join(
        base, dt.strftime('%Y'), dt.strftime('%m'), dt.strftime('%d'),
        dt.strftime('%H%M%S') + f'_{channel}.npy'
    )


def get_history_times(current_time, num_frames, stride_minutes):
    start = current_time - (num_frames - 1) * timedelta(minutes=stride_minutes)
    return [start + timedelta(minutes=stride_minutes) * i for i in range(num_frames)]


def get_target_times(current_time, horizons_minutes):
    return [current_time + timedelta(minutes=h) for h in horizons_minutes]


def all_files_exist(paths):
    return all(os.path.exists(p) for p in paths)


def is_valid_sample(radar_base, sat_base, history_times, target_times, channels):
    if not all_files_exist([radar_path(radar_base, t) for t in history_times]):
        return False
    for ch in channels:
        if not all_files_exist([satellite_path(sat_base, t, ch) for t in history_times]):
            return False
    if not all_files_exist([radar_path(radar_base, t) for t in target_times]):
        return False
    return True


def compute_patch_indices(grid_rows, grid_cols, patch_h, patch_w, stride,
                           start_row=0, start_col=0):
    patches = []
    r = start_row
    while r + patch_h <= grid_rows:
        c = start_col
        while c + patch_w <= grid_cols:
            patches.append((r, c))
            c += stride
        r += stride
    return patches


def _count_valid_patches_for_timepoint(args):
    (current_time, radar_base, sat_base, channels, num_in_frames,
     stride_minutes, horizons_minutes, patch_indices, patch_height,
     patch_width, nan_threshold, summer_only) = args

    if summer_only and current_time.month not in {6, 7, 8, 9}:
        return (None, 0)

    history_times = get_history_times(current_time, num_in_frames, stride_minutes)
    target_times  = get_target_times(current_time, horizons_minutes)

    if not is_valid_sample(radar_base, sat_base, history_times, target_times, channels):
        return (None, 0)

    ts_str = current_time.strftime('%Y-%m-%d %H:%M:%S')

    try:
        all_radar_frames = np.stack([
            np.load(radar_path(radar_base, t)).astype(np.float32)
            for t in history_times + target_times
        ], axis=0)
    except Exception:
        return (None, 0)

    out_frames = all_radar_frames[len(history_times):]
    in_frames  = all_radar_frames[:len(history_times)]

    count = 0
    for (pi, pj) in patch_indices:
        patch_slice = (slice(pi, pi + patch_height), slice(pj, pj + patch_width))

        out_patch = out_frames[(slice(None),) + patch_slice]
        out_nan_ratio = float(np.isnan(out_patch).mean())

        in_patch = in_frames[(slice(None),) + patch_slice]
        in_nan_ratio = float(np.isnan(in_patch).mean())

        nan_ratio = max(out_nan_ratio, in_nan_ratio)
        if nan_ratio < nan_threshold:
            count += 1

    return (ts_str, count)


def verify_count(
    radar_base, sat_base, start_date, end_date, channels,
    num_in_frames, stride_minutes, horizons_minutes,
    patch_height, patch_width, stride_patch,
    grid_rows, grid_cols, start_row, start_col,
    nan_threshold, summer_only, t_jump_minutes, max_workers,
    split_label="train",
    out_timestamps_csv=None,
    out_json=None,
):
    print("=" * 70)
    print("INDEPENDENT VERIFICATION -- recomputing exact patch count")
    print("=" * 70)
    print(f"Split label:   {split_label}")
    print(f"Date range:    {start_date} -> {end_date}")
    print(f"Summer only:   {summer_only}")
    print(f"nan_threshold: {nan_threshold}")
    print("=" * 70)

    patch_indices = compute_patch_indices(
        grid_rows, grid_cols, patch_height, patch_width, stride_patch,
        start_row, start_col,
    )
    n_patch_positions = len(patch_indices)
    print(f"Patch positions: {n_patch_positions}")

    start_dt = datetime.strptime(start_date, '%Y-%m-%d %H:%M')
    end_dt   = datetime.strptime(end_date,   '%Y-%m-%d %H:%M')
    timepoints = []
    t = start_dt
    while t <= end_dt:
        if (not summer_only) or (t.month in {6, 7, 8, 9}):
            timepoints.append(t)
        t += timedelta(minutes=t_jump_minutes)

    n_scanned = len(timepoints)
    print(f"Timepoints to scan (already JJAS-filtered if summer_only=True): {n_scanned:,}")
    print("=" * 70)

    packed_args = [
        (t, radar_base, sat_base, channels, num_in_frames, stride_minutes,
         horizons_minutes, patch_indices, patch_height, patch_width,
         nan_threshold, summer_only)
        for t in timepoints
    ]

    total_valid_patches = 0
    valid_timestamps = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for ts_str, count in tqdm(
            executor.map(_count_valid_patches_for_timepoint, packed_args, chunksize=4),
            total=len(packed_args),
            desc="Verifying",
        ):
            total_valid_patches += count
            if ts_str is not None:
                valid_timestamps.append(ts_str)

    n_valid_timepoints = len(valid_timestamps)
    avg_patches_per_valid_tp = (total_valid_patches / n_valid_timepoints
                                 if n_valid_timepoints > 0 else 0.0)
    avg_pct_of_max = (avg_patches_per_valid_tp / n_patch_positions * 100
                       if n_patch_positions > 0 else 0.0)

    print()
    print(f"VALID MULTI-FRAME TIMEPOINTS (\"common timestamps\", strict 8-frame check): {n_valid_timepoints:,}")
    print(f"TOTAL VALID PATCHES (independently recomputed):                          {total_valid_patches:,}")
    print(f"Average patches passing per valid timepoint: {avg_patches_per_valid_tp:.6f} / {n_patch_positions}")
    print()
    print("Compare TOTAL VALID PATCHES to the value reported by patches_counts_stat.py.")
    print("If they match exactly, the original count is confirmed correct.")
    print("=" * 70)

    if out_timestamps_csv:
        import pandas as pd
        os.makedirs(os.path.dirname(out_timestamps_csv) or '.', exist_ok=True)
        pd.DataFrame({'datetime': valid_timestamps}).to_csv(out_timestamps_csv, index=False)
        print(f"Saved valid timestamps list: {out_timestamps_csv}")

    # ------------------------------------------------------------------
    # JSON OUTPUT -- schema matches dataset_partitioning_summary.json
    # (generation_config + a "<split>_verification" block)
    # ------------------------------------------------------------------
    if out_json:
        result = {
            "generation_config": {
                "radar_base": radar_base,
                "sat_base": sat_base,
                "channels": channels,
                "num_in_frames": num_in_frames,
                "stride_minutes": stride_minutes,
                "horizons_minutes": horizons_minutes,
                "t_jump_minutes": t_jump_minutes,
                "patch_height": patch_height,
                "patch_width": patch_width,
                "stride_patch": stride_patch,
                "grid_rows": grid_rows,
                "grid_cols": grid_cols,
                "nan_threshold": nan_threshold,
                "patch_positions_per_timepoint": n_patch_positions,
                "summer_only_months": [6, 7, 8, 9] if summer_only else None,
            },
            f"{split_label}_verification": {
                "date_range": {"start": start_date, "end": end_date},
                "jjas_timepoints_scanned": n_scanned,
                "valid_multiframe_timepoints": n_valid_timepoints,
                "total_valid_patches": total_valid_patches,
                "average_patches_per_valid_timepoint": avg_patches_per_valid_tp,
                "average_patches_per_valid_timepoint_pct_of_max": round(avg_pct_of_max, 2),
                "verification_method": "independent from-scratch recomputation of patches_counts_stat.py logic",
            },
        }
        os.makedirs(os.path.dirname(out_json) or '.', exist_ok=True)
        with open(out_json, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Saved JSON summary: {out_json}")

    return n_valid_timepoints, total_valid_patches


def get_parser():
    parser = argparse.ArgumentParser(
        description="Independently verify the patch-count math from the original generation script"
    )
    parser.add_argument('--radar_base', type=str,
        default='/home/fe/sajib/scratch/weather-data/radar_de')
    parser.add_argument('--sat_base', type=str,
        default='/home/fe/sajib/scratch/weather-data/satellite_de_regridded')
    parser.add_argument('--start_date', type=str, default='2015-01-01 00:00')
    parser.add_argument('--end_date',   type=str, default='2022-12-31 23:55')
    parser.add_argument('--channels', nargs='+', default=['CH7', 'CH9'])
    parser.add_argument('--num_in_frames',   type=int, default=4)
    parser.add_argument('--stride_minutes',  type=int, default=5)
    parser.add_argument('--horizons', nargs='+', type=int, default=[15, 30, 45, 60])
    parser.add_argument('--t_jump',          type=int, default=5)
    parser.add_argument('--patch_height', type=int, default=256)
    parser.add_argument('--patch_width',  type=int, default=256)
    parser.add_argument('--stride_patch', type=int, default=64)
    parser.add_argument('--grid_rows',    type=int, default=1100)
    parser.add_argument('--grid_cols',    type=int, default=900)
    parser.add_argument('--start_row',    type=int, default=0)
    parser.add_argument('--start_col',    type=int, default=0)
    parser.add_argument('--nan_threshold', type=float, default=0.3)
    parser.add_argument('--summer_only', dest='summer_only', action='store_true', default=True,
        help="Restrict to JJAS (Jun-Sep) months. Default: True.")
    parser.add_argument('--no_summer_only', dest='summer_only', action='store_false',
        help="Disable the JJAS restriction and process the full year instead.")
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--split_label', type=str, default='train',
        help="Label used as the JSON key prefix, e.g. 'train', 'val', 'test'.")
    parser.add_argument('--out_timestamps_csv', type=str, default=None,
        help="Optional: save the list of valid multi-frame timestamps to this CSV")
    parser.add_argument('--out_json', type=str, default=None,
        help="Optional: save a JSON summary (schema matches dataset_partitioning_summary.json)")

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    verify_count(
        radar_base       = args.radar_base,
        sat_base         = args.sat_base,
        start_date       = args.start_date,
        end_date         = args.end_date,
        channels         = args.channels,
        num_in_frames    = args.num_in_frames,
        stride_minutes   = args.stride_minutes,
        horizons_minutes = args.horizons,
        patch_height     = args.patch_height,
        patch_width      = args.patch_width,
        stride_patch     = args.stride_patch,
        grid_rows        = args.grid_rows,
        grid_cols        = args.grid_cols,
        start_row        = args.start_row,
        start_col        = args.start_col,
        nan_threshold    = args.nan_threshold,
        summer_only      = args.summer_only,
        t_jump_minutes   = args.t_jump,
        max_workers      = args.workers,
        split_label      = args.split_label,
        out_timestamps_csv = args.out_timestamps_csv,
        out_json         = args.out_json,
    )