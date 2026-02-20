"""
Compute CHANNEL_STATS (mean & std) from satellite data for z-score normalization.

No clipping is applied — for convective storm prediction we want the full
dynamic range of satellite brightness temperatures preserved, especially
the cold cloud tops that indicate deep convection.

Invalid pixels (NaN, Inf, fill_value, zero) are excluded before computing stats.

Output example:
    CHANNEL_STATS = {
        7: {'mean': 29.96, 'std': 18.50},
        9: {'mean': 53.18, 'std': 30.78},
    }

Usage
-----
    python compute_channel_stats.py \
        --metadata   metadata/train_sampled.csv \
        --config     config/config.yml \
        --max_samples 100000 \
        --num_workers 8
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.path_utils import satellite_file_paths
from utils.io_utils import load_satellite_channel
from utils.time_utils import get_satellite_history_and_radar_target
from utils.config_loader import load_config


def load_one_satellite_file(root, timestamp, channel, nodata_value, fill_value):
    """
    Load a single satellite file and return valid pixel values only.

    Excludes:
        - NaN / Inf  (corrupted values)
        - fill_value (nodata replacement)
        - zeros      (invalid/masked pixels)
    """
    numpy_path, numpy_zip_path = satellite_file_paths(root, timestamp, channel)
    try:
        arr = load_satellite_channel(
            numpy_path, numpy_zip_path, nodata_value, fill_value
        )
        arr = arr.astype(np.float32).ravel()
        arr = arr[np.isfinite(arr)]  # drop NaN / Inf
        arr = arr[arr != fill_value]  # drop fill_value pixels
        arr = arr[arr > 0.0]  # drop zero / invalid pixels
        return arr if len(arr) > 0 else None
    except Exception:
        return None


def collect_pixels_for_channel(rows, channel, config, max_samples, num_workers):
    """
    Collect valid pixel values for one satellite channel across many files.

    Only the most-recent timestep of each sample is used to avoid loading
    redundant frames from the same scene.

    Parameters
    ----------
    rows : list of pd.Series
        Shuffled metadata rows.
    channel : int
        Satellite channel number.
    config : object
        Loaded YAML config.
    max_samples : int
        Maximum number of files to load.
    num_workers : int
        Thread-pool size for parallel I/O.

    Returns
    -------
    np.ndarray
        1-D float32 array of all valid pixel values.
    """
    satellite_root = config.paths.satellite_root
    nodata_value = config.satellite.nodata_value
    fill_value = config.satellite.fill_value
    cadence = config.satellite.cadence_minutes
    history_min = config.temporal.history_minutes
    lead_min = config.temporal.radar_lead_minutes
    dt_format = config.metadata.datetime_format

    # Build task list — one file per sample (most-recent timestep)
    tasks = []
    for row in rows[:max_samples]:
        sample_time = datetime.strptime(row["reference_time"], dt_format)
        satellite_times, _ = get_satellite_history_and_radar_target(
            sample_time, cadence, history_min, lead_min
        )
        t = satellite_times[-1]  # most-recent timestep
        tasks.append((satellite_root, t, channel, nodata_value, fill_value))

    # Parallel I/O
    all_pixels = []
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(load_one_satellite_file, *task): task for task in tasks}
        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"  Ch{channel}",
            ncols=100,
            leave=False,
        ):
            result = fut.result()
            if result is not None:
                all_pixels.append(result)

    if not all_pixels:
        raise RuntimeError(f"No valid pixels found for channel {channel}!")

    return np.concatenate(all_pixels)


def main():
    parser = argparse.ArgumentParser(
        description="Compute mean & std for satellite channel z-score normalization"
    )
    parser.add_argument(
        "--metadata",
        default="metadata/train_sampled.csv",
        help="Path to training metadata CSV",
    )
    parser.add_argument(
        "--config", default="config/config.yml", help="Path to main config YAML"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=100_000,
        help="Files to load per channel — 100k is statistically stable for millions of samples (default: 100000)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Parallel I/O threads (default: 8)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for shuffling (default: 42)"
    )
    parser.add_argument(
        "--out_json",
        default="metadata/channel_stats.json",
        help="Output JSON path (default: metadata/channel_stats.json)",
    )
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  CHANNEL STATS COMPUTATION  (mean & std only — no clipping)")
    print(f"{'='*70}")
    print(f"  Metadata    : {args.metadata}")
    print(f"  Config      : {args.config}")
    print(f"  Max samples : {args.max_samples:,} per channel")
    print(f"  Workers     : {args.num_workers}")
    print(f"  Seed        : {args.seed}")
    print(f"{'='*70}\n")

    config = load_config(args.config)
    channels = config.satellite.channels

    df = pd.read_csv(args.metadata)
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    rows = [df.iloc[i] for i in range(len(df))]

    print(f"  Total metadata rows : {len(df):,}")
    print(f"  Channels            : {channels}")
    print(
        f"  Effective files     : min({args.max_samples:,}, {len(df):,}) = {min(args.max_samples, len(df)):,} per channel\n"
    )

    # Compute stats
    channel_stats = {}

    for ch in channels:
        print(f"  ── Channel {ch} {'─'*50}")

        pixels = collect_pixels_for_channel(
            rows, ch, config, args.max_samples, args.num_workers
        )

        mean = float(np.mean(pixels))
        std = float(np.std(pixels))

        # Also print min/max/percentiles as sanity check (NOT used for normalization)
        p01 = float(np.percentile(pixels, 1))
        p99 = float(np.percentile(pixels, 99))
        vmin = float(pixels.min())
        vmax = float(pixels.max())

        channel_stats[ch] = {
            "mean": round(mean, 4),
            "std": round(std, 4),
        }

        print(f"    Valid pixels : {len(pixels):,}")
        print(f"    mean         : {mean:.4f}   ← use for z-score normalization")
        print(f"    std          : {std:.4f}   ← use for z-score normalization")
        print(f"    --- sanity check (not used) ---")
        print(f"    min          : {vmin:.4f}")
        print(f"    p1           : {p01:.4f}")
        print(f"    p99          : {p99:.4f}")
        print(f"    max          : {vmax:.4f}")
        print()

    # Print ready-to-paste dict
    print(f"{'='*70}")
    print("  CHANNEL_STATS  —  copy-paste into satellite_radar_dataset.py")
    print(f"{'='*70}")
    print("CHANNEL_STATS = {")
    for ch, stats in channel_stats.items():
        print(f"    {ch}: {{'mean': {stats['mean']}, 'std': {stats['std']}}},")
    print("}")
    print(f"{'='*70}\n")

    # Save JSON
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(channel_stats, f, indent=4)

    print(f"  Saved → {args.out_json}")
    print("  Done!\n")


if __name__ == "__main__":
    main()
