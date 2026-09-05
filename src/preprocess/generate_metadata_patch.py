import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import argparse

def radar_path(base, dt):
    """Build radar file path from datetime."""
    return os.path.join(
        base,
        dt.strftime('%y%m'),
        dt.strftime('%d'),
        dt.strftime('%y%m%d_%H%M') + '.npy'
    )


def satellite_path(base, dt, channel):
    """Build satellite file path from datetime and channel (CH7 or CH9)."""
    return os.path.join(
        base,
        dt.strftime('%Y'),
        dt.strftime('%m'),
        dt.strftime('%d'),
        dt.strftime('%H%M%S') + f'_{channel}.npy'
    )


def get_history_times(current_time, num_frames, stride_minutes):
    """
    Get list of history timestamps ending at current_time.
    current_time is the LAST input frame.
    """
    start = current_time - (num_frames - 1) * timedelta(minutes=stride_minutes)
    return [start + timedelta(minutes=stride_minutes) * i for i in range(num_frames)]


def get_target_times(current_time, horizons_minutes):
    """
    Get list of target timestamps from current_time.
    horizons_minutes: e.g. [15, 30, 45, 60]
    """
    return [current_time + timedelta(minutes=h) for h in horizons_minutes]



def all_files_exist(paths):
    """Check if all files in list exist."""
    return all(os.path.exists(p) for p in paths)


def is_valid_sample(
    radar_base, sat_base,
    history_times, target_times,
    channels
):
    """
    Check all required files exist:
    - Radar history frames
    - Satellite history frames (all channels)
    - Radar target frames (all horizons)
    """
    radar_history_paths = [radar_path(radar_base, t) for t in history_times]
    if not all_files_exist(radar_history_paths):
        return False

    for ch in channels:
        sat_paths = [satellite_path(sat_base, t, ch) for t in history_times]
        if not all_files_exist(sat_paths):
            return False

    radar_target_paths = [radar_path(radar_base, t) for t in target_times]
    if not all_files_exist(radar_target_paths):
        return False

    return True


def compute_patch_indices(
    grid_rows, grid_cols,
    patch_height, patch_width,
    stride,
    start_row=0, start_col=0,
):
    patches = []
    r = start_row
    while r + patch_height <= grid_rows:
        c = start_col
        while c + patch_width <= grid_cols:
            patches.append((r, c))
            c += stride
        r += stride
    return patches


CLIP_MAX = 128.0
SOFTLOG_EPS = 1.0


def softlog(x):
    return np.log(x + SOFTLOG_EPS)


def compute_patch_p99(patch):
    work = patch.copy().astype(np.float32)
    work = np.clip(work, 0.0, CLIP_MAX)
    work[np.isnan(work)] = 0.0
    return float(np.percentile(work, 99))


def compute_nan_ratio(arr):
    """Compute NaN ratio of array."""
    return float(np.isnan(arr).mean())


def _process_timepoint(args):
    (
        current_time,
        split_name,
        radar_base,
        sat_base,
        channels,
        num_in_frames,
        stride_minutes,
        horizons_minutes,
        patch_indices,
        patch_height,
        patch_width,
        nan_threshold,
        summer_only,
    ) = args

    if summer_only and current_time.month not in {6, 7, 8, 9}:
        return []

    history_times = get_history_times(current_time, num_in_frames, stride_minutes)
    target_times  = get_target_times(current_time, horizons_minutes)

    if not is_valid_sample(radar_base, sat_base, history_times, target_times, channels):
        return []

    try:
        all_radar_frames = np.stack([
            np.load(radar_path(radar_base, t)).astype(np.float32)
            for t in history_times + target_times
        ], axis=0) 
    except Exception:
        return []

    out_frames = all_radar_frames[len(history_times):]

    rows = []
    for (pi, pj) in patch_indices:
        patch_slice = (slice(pi, pi + patch_height), slice(pj, pj + patch_width))

        out_patch = out_frames[(slice(None),) + patch_slice]
        out_nan_ratio = float(np.isnan(out_patch).mean())

        in_patch = all_radar_frames[:len(history_times)][(slice(None),) + patch_slice]
        in_nan_ratio = float(np.isnan(in_patch).mean())

        nan_ratio = max(out_nan_ratio, in_nan_ratio)
        if nan_ratio >= nan_threshold:
            continue

        all_patch = all_radar_frames[(slice(None),) + patch_slice]  # [T_total, H, W]
        p99_val = compute_patch_p99(all_patch)

        row = {
            'datetime':   current_time.strftime('%Y-%m-%d %H:%M:%S'),
            'split':      split_name,
            'patch_row':  pi,
            'patch_col':  pj,
            'p99(x_seq)': round(p99_val, 4),
        }

        for h, t in zip(horizons_minutes, target_times):
            row[f'radar_time_{h}'] = t.strftime('%Y-%m-%d %H:%M:%S')
            row[f'radar_path_{h}'] = radar_path(radar_base, t)

        rows.append(row)

    return rows


def generate_patch_metadata(
    radar_base,
    sat_base,
    out_dir,
    start_date,
    end_date,
    split_name,
    channels,
    num_in_frames,
    stride_minutes,
    horizons_minutes,
    patch_height,
    patch_width,
    stride_patch,
    grid_rows,
    grid_cols,
    start_row,
    start_col,
    nan_threshold,
    summer_only,
    max_workers,
    t_jump_minutes,
):
    os.makedirs(out_dir, exist_ok=True)

    season_tag = '_summer' if summer_only else '_all'
    out_csv = os.path.join(
        out_dir,
        f'{split_name}_patch_{patch_height}x{patch_width}_s{stride_patch}{season_tag}.csv'
    )

    print("=" * 70)
    print(f"PATCH METADATA GENERATION — {split_name.upper()}")
    print("=" * 70)
    print(f"Radar base:      {radar_base}")
    print(f"Satellite base:  {sat_base}")
    print(f"Date range:      {start_date} → {end_date}")
    print(f"Patch size:      {patch_height}x{patch_width}")
    print(f"Patch stride:    {stride_patch}")
    print(f"Input frames:    {num_in_frames} × {stride_minutes}min")
    print(f"Horizons:        {horizons_minutes} min")
    print(f"Channels:        {channels}")
    print(f"Summer only:     {summer_only}")
    print(f"NaN threshold:   {nan_threshold}")
    print(f"Workers:         {max_workers}")
    print(f"Output:          {out_csv}")
    print("=" * 70)

    patch_indices = compute_patch_indices(
        grid_rows, grid_cols,
        patch_height, patch_width,
        stride_patch,
        start_row, start_col,
    )
    print(f"Patch positions: {len(patch_indices)}")

    start_dt = datetime.strptime(start_date, '%Y-%m-%d %H:%M')
    end_dt   = datetime.strptime(end_date,   '%Y-%m-%d %H:%M')
    timepoints = []
    t = start_dt
    while t <= end_dt:
        timepoints.append(t)
        t += timedelta(minutes=t_jump_minutes)

    print(f"Timepoints:      {len(timepoints):,}")
    print("=" * 70)

    packed_args = [
        (
            t,
            split_name,
            radar_base,
            sat_base,
            channels,
            num_in_frames,
            stride_minutes,
            horizons_minutes,
            patch_indices,
            patch_height,
            patch_width,
            nan_threshold,
            summer_only,
        )
        for t in timepoints
    ]

    all_rows = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for rows in tqdm(
            executor.map(_process_timepoint, packed_args, chunksize=4),
            total=len(packed_args),
            desc=f"Processing {split_name}",
        ):
            all_rows.extend(rows)

    print(f"\nValid patch samples: {len(all_rows):,}")

    if len(all_rows) == 0:
        print("WARNING: No valid samples found! Check paths and date range.")
        return None

    df = pd.DataFrame(all_rows)

    base_cols     = ['datetime', 'split', 'patch_row', 'patch_col', 'p99(x_seq)']
    horizon_cols  = []
    for h in horizons_minutes:
        horizon_cols += [f'radar_time_{h}', f'radar_path_{h}']

    df = df[base_cols + horizon_cols]

    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv} ({len(df):,} rows)")
    print("=" * 70)

    return df

def get_parser():
    parser = argparse.ArgumentParser(
        description="Generate patch-based metadata for nowcasting"
    )

    parser.add_argument('--radar_base', type=str,
        default='/home/fe/sajib/scratch/weather-data/radar_de')
    parser.add_argument('--sat_base', type=str,
        default='/home/fe/sajib/scratch/weather-data/satellite_de_regridded')
    parser.add_argument('--out_dir', type=str,
        default='metadata_patch')

    parser.add_argument('--splits', nargs='+',
        default=['train', 'val', 'test'],
        help="Splits to generate: train val test")

    parser.add_argument('--train_start', type=str, default='2015-01-01 00:00')
    parser.add_argument('--train_end',   type=str, default='2021-12-31 23:55')
    parser.add_argument('--val_start',   type=str, default='2022-01-01 00:00')
    parser.add_argument('--val_end',     type=str, default='2022-12-31 23:55')
    parser.add_argument('--test_start',  type=str, default='2023-01-01 00:00')
    parser.add_argument('--test_end',    type=str, default='2023-12-31 23:55')
    parser.add_argument('--channels', nargs='+', default=['CH7', 'CH9'])
    parser.add_argument('--num_in_frames',   type=int, default=4)
    parser.add_argument('--stride_minutes',  type=int, default=5)
    parser.add_argument('--horizons', nargs='+', type=int, default=[15, 30, 45, 60])
    parser.add_argument('--t_jump',          type=int, default=5,
        help="Minutes between sampled timepoints")
    parser.add_argument('--patch_height', type=int, default=128)
    parser.add_argument('--patch_width',  type=int, default=128)
    parser.add_argument('--stride_patch', type=int, default=64,
        help="Stride between patches (50% overlap default)")
    parser.add_argument('--grid_rows',    type=int, default=1100)
    parser.add_argument('--grid_cols',    type=int, default=900)
    parser.add_argument('--start_row',    type=int, default=0)
    parser.add_argument('--start_col',    type=int, default=0)
    parser.add_argument('--nan_threshold', type=float, default=0.3,
        help="Reject patches with NaN ratio >= this value")
    parser.add_argument('--summer_only', action='store_true',
        help="Only process Jun/Jul/Aug/Sept")
    parser.add_argument('--workers', type=int, default=12)

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args   = parser.parse_args()
    date_ranges = {
        'train': (args.train_start, args.train_end),
        'val':   (args.val_start,   args.val_end),
        'test':  (args.test_start,  args.test_end),
    }

    for split in args.splits:
        start_date, end_date = date_ranges[split]
        generate_patch_metadata(
            radar_base       = args.radar_base,
            sat_base         = args.sat_base,
            out_dir          = args.out_dir,
            start_date       = start_date,
            end_date         = end_date,
            split_name       = split,
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
            max_workers      = args.workers,
            t_jump_minutes   = args.t_jump,
        )