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
    numpy_path, numpy_zip_path = satellite_file_paths(root, timestamp, channel)
    try:
        arr = load_satellite_channel(
            numpy_path, numpy_zip_path, nodata_value, fill_value
        )
        arr = arr.astype(np.float32).ravel()
        arr = arr[np.isfinite(arr)]  
        arr = arr[arr != fill_value] 
        arr = arr[arr > 0.0]  
        return arr if len(arr) > 0 else None
    except Exception:
        return None


def collect_pixels_for_channel(rows, channel, config, max_samples, num_workers):
    satellite_root = config.paths.satellite_root
    nodata_value = config.satellite.nodata_value
    fill_value = config.satellite.fill_value
    cadence = config.satellite.cadence_minutes
    history_min = config.temporal.history_minutes
    lead_min = config.temporal.radar_lead_minutes
    dt_format = config.metadata.datetime_format

    tasks = []
    for row in rows[:max_samples]:
        sample_time = datetime.strptime(row["reference_time"], dt_format)
        satellite_times, _ = get_satellite_history_and_radar_target(
            sample_time, cadence, history_min, lead_min
        )
        t = satellite_times[-1]
        tasks.append((satellite_root, t, channel, nodata_value, fill_value))

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
        default="metadata/train_multihorizon.csv",
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

    channel_stats = {}

    for ch in channels:
        print(f"  ── Channel {ch} {'─'*50}")

        pixels = collect_pixels_for_channel(
            rows, ch, config, args.max_samples, args.num_workers
        )

        mean = float(np.mean(pixels))
        std = float(np.std(pixels))

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

    print(f"{'='*70}")
    print("  CHANNEL_STATS  —  copy-paste into satellite_radar_dataset.py")
    print(f"{'='*70}")
    print("CHANNEL_STATS = {")
    for ch, stats in channel_stats.items():
        print(f"    {ch}: {{'mean': {stats['mean']}, 'std': {stats['std']}}},")
    print("}")
    print(f"{'='*70}\n")

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(channel_stats, f, indent=4)

    print(f"  Saved → {args.out_json}")
    print("  Done!\n")


if __name__ == "__main__":
    main()
