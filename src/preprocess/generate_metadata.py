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


"""
Metadata Generation and Dataset Splitting Pipeline.

This module generates per-sample metadata for a satellite–radar dataset and
automatically splits the dataset into train/validation/test CSV files.

For each reference timestamp:
    1. Computes satellite history timestamps and radar target timestamp.
    2. Validates availability of required satellite channels and radar file.
    3. Loads and preprocesses the radar frame (clipping).
    4. Computes statistical metadata (e.g., percentiles).
    5. Computes NaN statistics.
    6. Stores all metadata in a CSV file.

The final output consists of:
    - metadata/all_samples.csv
    - metadata/train.csv
    - metadata/val.csv
    - metadata/test.csv

The pipeline supports multiprocessing for faster metadata generation.

Typical usage (CLI):
    python generate_metadata.py --config config/config.yml --workers 8
"""

def _process_one_timestamp(args):
    """
    Process a single reference timestamp and compute metadata.

    This function:
        - Computes satellite history timestamps and radar lead target time.
        - Validates availability of required satellite channels and radar file.
        - Loads and preprocesses the radar frame.
        - Computes statistical metadata (e.g., percentiles).
        - Computes NaN statistics.
        - Returns a dictionary suitable for CSV writing.

    Parameters
    ----------
    args : tuple
        A tuple containing:
            current_time : datetime
                Reference timestamp.
            split_name : str
                Dataset split name ("train", "val", or "test").
            config : object
                Configuration object loaded from YAML.

    Returns
    -------
    dict or None
        Dictionary containing metadata fields if valid, otherwise None.

        Returned dictionary contains:
            - split
            - reference_time
            - radar_time
            - radar_path
            - valid_cells
            - nan_cells
            - nan_ratio
            - all_nan
            - percentile statistics (if computed)
    """

    current_time, split_name, config = args

    satellite_times, radar_time = get_satellite_history_and_radar_target(
        current_time,
        config.satellite.cadence_minutes,
        config.temporal.history_minutes,
        config.temporal.radar_lead_minutes,
    )

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

    radar = clip(radar, 0.0, config.transform.radar.clip_max)

    stats = compute_metadata(
        radar,
        config.metadata.percentiles,
    )

    valid_mask = np.isfinite(radar)

    nan_info = {
        "valid_cells": int(valid_mask.sum()),
        "nan_cells": int((~valid_mask).sum()),
        "nan_ratio": float((~valid_mask).mean()),
        "all_nan": bool(valid_mask.sum() == 0),
    }

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


def generate_metadata_and_split(
    config_path,
    force=False,
    num_workers=None,
):
    
    """
    Generate metadata CSV files and create train/val/test splits.

    This function:
        1. Loads the configuration file.
        2. Iterates over all timestamps in each split.
        3. Uses multiprocessing to compute metadata.
        4. Writes a combined CSV (all_samples.csv).
        5. Splits it into train/val/test CSV files.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.
    force : bool, optional
        If True, regenerate metadata even if output already exists.
        Default is False.
    num_workers : int or None, optional
        Number of multiprocessing workers.
        If None, uses (CPU count - 1).

    Output
    ------
    Creates a "metadata" directory containing:
        - all_samples.csv
        - train.csv
        - val.csv
        - test.csv

    Notes
    -----
    - Multiprocessing significantly speeds up metadata generation.
    - Invalid samples (missing files, corrupted radar frames, etc.)
      are automatically skipped.
    - Splits are defined in the configuration file.
    """

    out_dir = "metadata"

    os.makedirs(out_dir, exist_ok=True)

    all_csv = os.path.join(out_dir, "all_samples.csv")
    train_csv = os.path.join(out_dir, "train.csv")
    val_csv = os.path.join(out_dir, "val.csv")
    test_csv = os.path.join(out_dir, "test.csv")

    if os.path.exists(all_csv) and not force:
        print("Metadata already exists – skipping generation")
        return

    config = load_config(config_path)

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

    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    print("=" * 80)
    print("GENERATING METADATA")
    print("=" * 80)
    print(f"Workers        : {num_workers}")
    print(f"Total samples  : {len(tasks)}")
    print("=" * 80)

    with Pool(processes=num_workers) as pool:
        results = pool.map(_process_one_timestamp, tasks)

    rows = [r for r in results if r is not None]

    print(f"Valid samples: {len(rows)}")

    write_metadata_csv(rows, all_csv)

    print(f"Saved: {all_csv}")

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
