from pathlib import Path
from datetime import datetime
import pandas as pd


RADAR_ROOT = Path("~/scratch/weather-data/radar_de").expanduser()
SATELLITE_ROOT = Path("~/scratch/weather-data/satellite_de_regridded").expanduser() 

PATCHES_PER_IMAGE = 154

TRAINING_YEARS = range(2015, 2023)
SUMMER_MONTHS = {6, 7, 8, 9}
VALIDATION_YEARS = {2023}
TEST_YEARS = {2024}

OUTPUT_DIR = Path("./timestamp_check_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_timestamp_radar(file_path):
    """
    Radar filename format: YYMMDD_HHMM.npy
    Example: 150101_0000.npy -> 2015-01-01 00:00
    """
    try:
        return datetime.strptime(file_path.stem, "%y%m%d_%H%M")
    except ValueError:
        return None


def extract_timestamp_satellite(file_path):
    """
    Returns (datetime, channel) or (None, None) if it doesn't match.
    """
    try:
        parts = file_path.parts
        yyyy, mm, dd = parts[-4], parts[-3], parts[-2]
        stem = file_path.stem  # e.g. '000000_CH7'
        time_str, channel = stem.rsplit("_", 1)  # '000000', 'CH7'
        dt = datetime.strptime(f"{yyyy}{mm}{dd}_{time_str[:4]}", "%Y%m%d_%H%M")  # truncate seconds
        return dt, channel
    except (ValueError, IndexError):
        return None, None


def get_radar_timestamps(root):
    timestamps = set()
    invalid_files = []
    files = sorted(root.rglob("*.npy"))
    print(f"\nScanning (radar): {root}")
    print(f"Found .npy files: {len(files):,}")
    for file_path in files:
        ts = extract_timestamp_radar(file_path)
        if ts is not None:
            timestamps.add(ts)
        else:
            invalid_files.append(str(file_path))
    return timestamps, invalid_files


def get_satellite_timestamps(root):
    """
    Only counts a timestamp as 'available' if BOTH CH7 and CH9 exist for it,
    since the model requires both channels together as input.
    """
    channel_map = {}  # timestamp -> set of channels found
    invalid_files = []
    files = sorted(root.rglob("*.npy"))
    print(f"\nScanning (satellite): {root}")
    print(f"Found .npy files: {len(files):,}")
    for file_path in files:
        ts, channel = extract_timestamp_satellite(file_path)
        if ts is not None and channel in ("CH7", "CH9"):
            channel_map.setdefault(ts, set()).add(channel)
        else:
            invalid_files.append(str(file_path))

    timestamps = {ts for ts, chans in channel_map.items() if {"CH7", "CH9"}.issubset(chans)}
    incomplete = {ts for ts, chans in channel_map.items() if not {"CH7", "CH9"}.issubset(chans)}
    if incomplete:
        print(f"WARNING: {len(incomplete):,} satellite timestamps have only ONE of CH7/CH9 (excluded).")

    return timestamps, invalid_files


def select_split(timestamps, years, summer_only=False):
    selected = set()
    for timestamp in timestamps:
        if timestamp.year not in years:
            continue
        if summer_only and timestamp.month not in SUMMER_MONTHS:
            continue
        selected.add(timestamp)
    return selected


print("=" * 70)
print("RADAR / SATELLITE TIMESTAMP CHECK")
print("=" * 70)
print(f"\nRadar directory: {RADAR_ROOT}")
print(f"Satellite directory: {SATELLITE_ROOT}")

radar_timestamps, radar_invalid = get_radar_timestamps(RADAR_ROOT)
print("\nRadar timestamp summary")
print("-" * 70)
print(f"Unique radar timestamps: {len(radar_timestamps):,}")
print(f"Invalid radar filenames: {len(radar_invalid):,}")

satellite_timestamps, satellite_invalid = get_satellite_timestamps(SATELLITE_ROOT)
print("\nSatellite timestamp summary")
print("-" * 70)
print(f"Unique satellite timestamps (both CH7+CH9 present): {len(satellite_timestamps):,}")
print(f"Invalid/unmatched satellite filenames: {len(satellite_invalid):,}")

common_timestamps = radar_timestamps.intersection(satellite_timestamps)
radar_only_timestamps = radar_timestamps - satellite_timestamps
satellite_only_timestamps = satellite_timestamps - radar_timestamps

print("\nOverall timestamp comparison")
print("-" * 70)
print(f"Radar timestamps:         {len(radar_timestamps):,}")
print(f"Satellite timestamps:     {len(satellite_timestamps):,}")
print(f"Common timestamps:        {len(common_timestamps):,}")
print(f"Radar-only timestamps:    {len(radar_only_timestamps):,}")
print(f"Satellite-only timestamps:{len(satellite_only_timestamps):,}")

splits = {
    "Training": {"years": set(TRAINING_YEARS), "summer_only": True},
    "Validation": {"years": VALIDATION_YEARS, "summer_only": True},
    "Test": {"years": TEST_YEARS, "summer_only": False},
}

results = []
for split_name, settings in splits.items():
    years = settings["years"]
    summer_only = settings["summer_only"]

    radar_split = select_split(radar_timestamps, years, summer_only)
    satellite_split = select_split(satellite_timestamps, years, summer_only)
    common_split = radar_split.intersection(satellite_split)
    radar_only_split = radar_split - satellite_split
    satellite_only_split = satellite_split - radar_split

    theoretical_patches_radar = len(radar_split) * PATCHES_PER_IMAGE
    theoretical_patches_satellite = len(satellite_split) * PATCHES_PER_IMAGE
    theoretical_patches_common = len(common_split) * PATCHES_PER_IMAGE

    results.append({
        "Split": split_name,
        "Radar Timesteps": len(radar_split),
        "Satellite Timesteps": len(satellite_split),
        "Common Timesteps": len(common_split),
        "Radar-only Timesteps": len(radar_only_split),
        "Satellite-only Timesteps": len(satellite_only_split),
        "Patches per Image": PATCHES_PER_IMAGE,
        "Theoretical Radar Patches": theoretical_patches_radar,
        "Theoretical Satellite Patches": theoretical_patches_satellite,
        "Theoretical Common Patches": theoretical_patches_common,
    })

    print("\n" + "=" * 70)
    print(split_name)
    print("=" * 70)
    print(f"Radar timesteps:                  {len(radar_split):,}")
    print(f"Satellite timesteps:              {len(satellite_split):,}")
    print(f"Common radar-satellite timesteps: {len(common_split):,}")
    print(f"Radar-only timesteps:             {len(radar_only_split):,}")
    print(f"Satellite-only timesteps:         {len(satellite_only_split):,}")
    print(f"Theoretical common patches:       {theoretical_patches_common:,}")

    pd.DataFrame({"timestamp": sorted(common_split)}).to_csv(
        OUTPUT_DIR / f"{split_name.lower()}_common_timestamps.csv", index=False)
    pd.DataFrame({"timestamp": sorted(radar_only_split)}).to_csv(
        OUTPUT_DIR / f"{split_name.lower()}_radar_only.csv", index=False)
    pd.DataFrame({"timestamp": sorted(satellite_only_split)}).to_csv(
        OUTPUT_DIR / f"{split_name.lower()}_satellite_only.csv", index=False)

summary_df = pd.DataFrame(results)
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(summary_df.to_string(index=False))

summary_file = OUTPUT_DIR / "dataset_timestamp_summary.csv"
summary_df.to_csv(summary_file, index=False)
print(f"\nSummary saved to: {summary_file}")

if radar_invalid:
    pd.DataFrame({"file": radar_invalid}).to_csv(
        OUTPUT_DIR / "invalid_radar_filenames.csv", index=False)
if satellite_invalid:
    pd.DataFrame({"file": satellite_invalid}).to_csv(
        OUTPUT_DIR / "invalid_satellite_filenames.csv", index=False)

expected_values = {"Training": 281088, "Validation": 35136, "Test": 105408}
print("\n" + "=" * 70)
print("COMPARISON WITH CURRENT THESIS TABLE")
print("=" * 70)
for _, row in summary_df.iterrows():
    split = row["Split"]
    actual = row["Common Timesteps"]
    expected = expected_values[split]
    difference = actual - expected
    print(f"{split:12s} | Expected: {expected:8,} | Common: {actual:8,} | Difference: {difference:+,}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
print(f"\nAll output files are saved in:\n{OUTPUT_DIR.resolve()}")