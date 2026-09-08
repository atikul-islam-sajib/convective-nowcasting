import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import argparse

def radar_path(base, dt):
    return os.path.join(
        base,
        dt.strftime('%y%m'),
        dt.strftime('%d'),
        dt.strftime('%y%m%d_%H%M') + '.npy'
    )


def satellite_path(base, dt, channel):
    return os.path.join(
        base,
        dt.strftime('%Y'),
        dt.strftime('%m'),
        dt.strftime('%d'),
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

CLIP_MAX    = 128.0
SOFTLOG_EPS = 1.0


def compute_patch_p99(arr):
    work = arr.copy().astype(np.float32)
    work = np.clip(work, 0.0, CLIP_MAX)
    work[np.isnan(work)] = 0.0
    return float(np.percentile(work, 99))

def _process_timepoint_fullimage(args):
    (
        current_time,
        split_name,
        radar_base,
        sat_base,
        channels,
        num_in_frames,
        stride_minutes,
        horizons_minutes,
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
        ], axis=0)  # [T_in + T_out, H, W]
    except Exception:
        return []

    p99_val = compute_patch_p99(all_radar_frames)

    row = {
        'datetime':   current_time.strftime('%Y-%m-%d %H:%M:%S'),
        'split':      split_name,
        'p99(x_seq)': round(p99_val, 4),
    }

    for h, t in zip(horizons_minutes, target_times):
        row[f'radar_time_{h}'] = t.strftime('%Y-%m-%d %H:%M:%S')
        row[f'radar_path_{h}'] = radar_path(radar_base, t)

    return [row]


def generate_fullimage_metadata(
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
    summer_only,
    max_workers,
    t_jump_minutes,
):
    os.makedirs(out_dir, exist_ok=True)

    season_tag = '_summer' if summer_only else '_all'
    out_csv = os.path.join(
        out_dir,
        f'{split_name}_fullimage{season_tag}.csv'
    )

    print("=" * 70)
    print(f"FULL IMAGE METADATA GENERATION -- {split_name.upper()}")
    print("=" * 70)
    print(f"Radar base:      {radar_base}")
    print(f"Satellite base:  {sat_base}")
    print(f"Date range:      {start_date} -> {end_date}")
    print(f"Input frames:    {num_in_frames} x {stride_minutes}min")
    print(f"Horizons:        {horizons_minutes} min")
    print(f"Channels:        {channels}")
    print(f"Summer only:     {summer_only}  (JJAS: Jun-Sep)")
    print(f"Workers:         {max_workers}")
    print(f"Output:          {out_csv}")
    print("=" * 70)

    # Build timepoints
    start_dt   = datetime.strptime(start_date, '%Y-%m-%d %H:%M')
    end_dt     = datetime.strptime(end_date,   '%Y-%m-%d %H:%M')
    timepoints = []
    t = start_dt
    while t <= end_dt:
        timepoints.append(t)
        t += timedelta(minutes=t_jump_minutes)

    print(f"Timepoints (raw scan, pre-summer-filter): {len(timepoints):,}")
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
            summer_only,
        )
        for t in timepoints
    ]

    all_rows = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for rows in tqdm(
            executor.map(_process_timepoint_fullimage, packed_args, chunksize=4),
            total=len(packed_args),
            desc=f"Processing {split_name}",
        ):
            all_rows.extend(rows)

    print(f"\nValid full-image samples: {len(all_rows):,}")

    if len(all_rows) == 0:
        print("WARNING: No valid samples found! Check paths and date range.")
        return None

    df = pd.DataFrame(all_rows)

    base_cols    = ['datetime', 'split', 'p99(x_seq)']
    horizon_cols = []
    for h in horizons_minutes:
        horizon_cols += [f'radar_time_{h}', f'radar_path_{h}']

    df = df[base_cols + horizon_cols]
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv} ({len(df):,} rows)")
    print("=" * 70)

    return df

def get_parser():
    parser = argparse.ArgumentParser(
        description="Generate full-image metadata for nowcasting test evaluation"
    )

    parser.add_argument('--radar_base', type=str,
        default='/home/fe/sajib/scratch/weather-data/radar_de')
    parser.add_argument('--sat_base', type=str,
        default='/home/fe/sajib/scratch/weather-data/satellite_de_regridded')
    parser.add_argument('--out_dir', type=str,
        default='metadata_patch')

    parser.add_argument('--splits', nargs='+', default=['test'])

    parser.add_argument('--train_start', type=str, default='2015-01-01 00:00')
    parser.add_argument('--train_end',   type=str, default='2022-12-31 23:55')
    parser.add_argument('--val_start',   type=str, default='2023-01-01 00:00')
    parser.add_argument('--val_end',     type=str, default='2023-12-31 23:55')
    parser.add_argument('--test_start',  type=str, default='2024-01-01 00:00')
    parser.add_argument('--test_end',    type=str, default='2024-12-31 23:55')

    parser.add_argument('--channels',       nargs='+', default=['CH7', 'CH9'])
    parser.add_argument('--num_in_frames',  type=int,  default=4)
    parser.add_argument('--stride_minutes', type=int,  default=5)
    parser.add_argument('--horizons',       nargs='+', type=int, default=[15, 30, 45, 60])
    parser.add_argument('--t_jump',         type=int,  default=5)
    parser.add_argument('--summer_only',    action='store_true')
    parser.add_argument('--workers',        type=int,  default=12)

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
        generate_fullimage_metadata(
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
            summer_only      = args.summer_only,
            max_workers      = args.workers,
            t_jump_minutes   = args.t_jump,
        )