import os
import numpy as np
import pandas as pd
from datetime import timedelta
from multiprocessing import Pool, cpu_count

from utils.config_loader import load_config
from utils.time_utils_multihorizon import get_satellite_history_and_radar_targets, compute_forecast_horizons
from utils.path_utils import radar_file_path
from utils.io_utils import load_radar_frame
from utils.validation_utils_multihorizon import is_valid_sample
from utils.transform_utils import clip, log_transform
from utils.metadata_utils import compute_metadata
from utils.metadata_writer import write_metadata_csv


def _process_one_timestamp(args):
    current_time, split_name, config = args
    satellite_times, radar_target_times = get_satellite_history_and_radar_targets(
        current_time,
        config.satellite.cadence_minutes,
        config.temporal.history_minutes,
        config.temporal.radar_lead_minutes,
    )

    if not is_valid_sample(
        config.paths.satellite_root,
        config.paths.radar_root,
        satellite_times,
        radar_target_times,  
        config.satellite.channels,
    ):
        return None

    horizons = compute_forecast_horizons(config.temporal.radar_lead_minutes)

    horizon_data = {}
    for h, radar_time in zip(horizons, radar_target_times):
        radar_path = radar_file_path(
            config.paths.radar_root,
            radar_time,
        )
        horizon_data[f'radar_time_{h}'] = radar_time.strftime(
            config.metadata.datetime_format
        )
        horizon_data[f'radar_path_{h}'] = radar_path

    last_radar_path = horizon_data[f'radar_path_{horizons[-1]}']

    try:
        radar = load_radar_frame(last_radar_path)
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
        **horizon_data,
        **nan_info,
        **(stats if stats is not None else {}),
    }


def generate_metadata_and_split(
    config_path,
    force=False,
    num_workers=None,
):
    out_dir = "metadata"
    os.makedirs(out_dir, exist_ok=True)

    all_csv   = os.path.join(out_dir, "all_samples_multihorizon_summer.csv")
    train_csv = os.path.join(out_dir, "train_multihorizon_summer.csv")
    val_csv   = os.path.join(out_dir, "val_multihorizon_summer.csv")
    test_csv  = os.path.join(out_dir, "test_multihorizon_summer.csv")

    if os.path.exists(all_csv) and not force:
        print("Multi-horizon summer metadata already exists – skipping generation")
        return

    config = load_config(config_path)

    horizons = compute_forecast_horizons(config.temporal.radar_lead_minutes)
    print("=" * 80)
    print("MULTI-HORIZON METADATA GENERATION — SUMMER ONLY (JJA)")
    print("=" * 80)
    print(f"radar_lead_minutes : {config.temporal.radar_lead_minutes}")
    print(f"Forecast horizons  : {horizons}")
    print(f"n_horizons         : {len(horizons)}")
    print(f"Summer months      : June, July, August (6, 7, 8)")
    print("=" * 80)

    SUMMER_MONTHS = {6, 7, 8}

    tasks = []
    cadence = config.satellite.cadence_minutes

    for split_name in ["train", "val", "test"]:
        split = getattr(config.splits, split_name)
        current_time = split.start

        while current_time <= split.end:
            if current_time.month in SUMMER_MONTHS:
                tasks.append((current_time, split_name, config))
            current_time += timedelta(minutes=cadence)

    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    print(f"Workers        : {num_workers}")
    print(f"Total tasks    : {len(tasks)}")
    print("=" * 80)

    with Pool(processes=num_workers) as pool:
        results = pool.map(_process_one_timestamp, tasks)

    rows = [r for r in results if r is not None]
    print(f"Valid samples: {len(rows)}")

    write_metadata_csv(rows, all_csv)
    print(f"Saved: {all_csv}")

    df = pd.read_csv(all_csv)

    df['month'] = pd.to_datetime(df['reference_time']).dt.month
    assert df['month'].isin(SUMMER_MONTHS).all(), "Non-summer samples found!"
    df = df.drop(columns=['month'])

    df_train = df[df["split"] == "train"]
    df_val   = df[df["split"] == "val"]
    df_test  = df[df["split"] == "test"]

    df_train.to_csv(train_csv, index=False)
    df_val.to_csv(val_csv,     index=False)
    df_test.to_csv(test_csv,   index=False)

    print(f"Saved: {train_csv} ({len(df_train)})")
    print(f"Saved: {val_csv}   ({len(df_val)})")
    print(f"Saved: {test_csv}  ({len(df_test)})")

    print("=" * 80)
    print("SUMMER METADATA GENERATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Generate multi-horizon metadata and split into train/val/test CSVs"
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