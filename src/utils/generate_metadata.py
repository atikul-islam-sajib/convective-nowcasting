import os
import numpy as np
import pandas as pd
from datetime import timedelta
from multiprocessing import Pool, cpu_count

from utils.config_loader import load_config
from utils.time_utils import get_satellite_history_and_radar_target
from utils.path_utils import radar_file_path
from utils.io_utils import load_radar_frame
from utils.validation_utils import is_valid_sample
from utils.transform_utils import clip, log_transform
from utils.metadata_utils import compute_metadata
from utils.metadata_writer import write_metadata_csv


# ============================================================
# WORKER (FOR MULTIPROCESSING)
# ============================================================

def _process_one_timestamp(args):

    current_time, split_name, config = args

    satellite_times, radar_time = get_satellite_history_and_radar_target(
        current_time,
        config.satellite.cadence_minutes,
        config.temporal.history_minutes,
        config.temporal.radar_lead_minutes,
    )

    # Fast availability check
    if not is_valid_sample(
        config.paths.satellite_root,
        config.paths.radar_root,
        satellite_times,
        radar_time,
        config.satellite.channels,
    ):
        return None

    radar_path = radar_file_path(
        config.paths.radar_root,
        radar_time,
    )

    try:
        radar = load_radar_frame(radar_path)
    except Exception:
        return None

    # Preprocess radar
    radar = clip(radar, 0.0, config.transform.radar.clip_max)

   # if config.transform.radar.log_transform:
   #     radar = log_transform(radar)
    # Statistics (p99, etc.)
    stats = compute_metadata(
        radar,
        config.metadata.percentiles,
    )
    # NaN info
    valid_mask = np.isfinite(radar)

    nan_info = {
        "valid_cells": int(valid_mask.sum()),
        "nan_cells": int((~valid_mask).sum()),
        "nan_ratio": float((~valid_mask).mean()),
        "all_nan": bool(valid_mask.sum() == 0),
    }

    # Statistics (p99, etc.)
   # stats = compute_metadata(
   #     radar,
   #     config.metadata.percentiles,
   # )

    return {
        "split": split_name,
        "reference_time": current_time.strftime(
            config.metadata.datetime_format
        ),
        "radar_time": radar_time.strftime(
            config.metadata.datetime_format
        ),
        "radar_path": radar_path,
        **nan_info,
        **(stats if stats is not None else {}),
    }


# MAIN FUNCTION

def generate_metadata_and_split(
    config_path,
    force=False,
    num_workers=None,
):

    out_dir = "metadata"

    os.makedirs(out_dir, exist_ok=True)

    all_csv = os.path.join(out_dir, "all_samples.csv")
    train_csv = os.path.join(out_dir, "train.csv")
    val_csv = os.path.join(out_dir, "val.csv")
    test_csv = os.path.join(out_dir, "test.csv")

    # Skip if already exists
    if os.path.exists(all_csv) and not force:
        print("Metadata already exists – skipping generation")
        return

    # Load config
    config = load_config(config_path)


    # Prepare tasks

    tasks = []

    cadence = config.satellite.cadence_minutes

    for split_name in ["train", "val", "test"]:

        split = getattr(config.splits, split_name)

        current_time = split.start

        while current_time <= split.end:

            tasks.append(
                (current_time, split_name, config)
            )

            current_time += timedelta(minutes=cadence)

    # Parallel setup

    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    print("=" * 80)
    print("GENERATING METADATA")
    print("=" * 80)
    print(f"Workers        : {num_workers}")
    print(f"Total samples  : {len(tasks)}")
    print("=" * 80)

    # Run multiprocessing

    with Pool(processes=num_workers) as pool:
        results = pool.map(_process_one_timestamp, tasks)

    # Keep valid rows
    rows = [r for r in results if r is not None]

    print(f"Valid samples: {len(rows)}")

    # Write ALL

    write_metadata_csv(rows, all_csv)

    print(f"Saved: {all_csv}")

    # Split into train / val / test

    df = pd.read_csv(all_csv)

    df_train = df[df["split"] == "train"]
    df_val = df[df["split"] == "val"]
    df_test = df[df["split"] == "test"]

    df_train.to_csv(train_csv, index=False)
    df_val.to_csv(val_csv, index=False)
    df_test.to_csv(test_csv, index=False)

    print(f"Saved: {train_csv} ({len(df_train)})")
    print(f"Saved: {val_csv}   ({len(df_val)})")
    print(f"Saved: {test_csv}  ({len(df_test)})")

    print("=" * 80)
    print("METADATA GENERATION COMPLETE")
    print("=" * 80)


# CLI

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Generate metadata and split into train/val/test CSVs"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yml",
        help="Path to config.yml"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of workers"
    )

    args = parser.parse_args()

    generate_metadata_and_split(
        config_path=args.config,
        force=args.force,
        num_workers=args.workers,
    )
