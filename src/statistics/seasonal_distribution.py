import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import csv

RADAR_DIR = Path("/home/fe/sajib/scratch/weather-data/radar_de")
YEARS     = list(range(2016, 2025))
MAX_MM_H  = 128.0
N_WORKERS = 16
MINUTES_PER_FILE = 5.0
HOURS_PER_FILE = MINUTES_PER_FILE / 60.0 

INVALID_SENTINELS = {-1.0, -9999.0, 9999.0}

SEASON_MAP = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3:  "Spring", 4: "Spring", 5: "Spring",
    6:  "Summer", 7: "Summer", 8: "Summer",
    9:  "Autumn", 10: "Autumn", 11: "Autumn",
}

print("Collecting files …")
all_files = []

for yymm_dir in sorted(RADAR_DIR.iterdir()):
    if not yymm_dir.is_dir():
        continue
    name = yymm_dir.name
    if len(name) != 4 or not name.isdigit():
        continue

    yy = int(name[:2])
    mm = int(name[2:])
    year  = 2000 + yy
    month = mm

    if year not in YEARS:
        continue
    if not (1 <= month <= 12):
        continue

    season_year = year + 1 if month == 12 else year
    if season_year not in YEARS:
        continue

    season = SEASON_MAP[month]

    for p in yymm_dir.rglob("*.npy"):
        all_files.append((season_year, season, p))

print(f"  Total files matched: {len(all_files):,}")

if len(all_files) == 0:
    print("ERROR: No files found. Check RADAR_DIR path.")
    raise SystemExit(1)

def load_mean(path: Path) -> float:
    try:
        arr = np.load(path, mmap_mode="r").astype(np.float32)

        valid = np.isfinite(arr)
        for sentinel in INVALID_SENTINELS:
            valid &= (arr != sentinel)
        valid &= (arr >= 0.0)

        n_valid = int(valid.sum())
        if n_valid < 100:
            return np.nan

        values = np.clip(arr[valid], 0.0, MAX_MM_H)
        return float(values.mean())

    except Exception:
        return np.nan


print("\nDiagnostic on first file:")
first_path = all_files[0][2]
arr_diag = np.load(first_path, mmap_mode="r").astype(np.float32)
print(f"  Path        : {first_path}")
print(f"  Shape       : {arr_diag.shape}")
print(f"  dtype       : {arr_diag.dtype}")
print(f"  Total pixels: {arr_diag.size:,}")
print(f"  np.nan count: {int(np.sum(~np.isfinite(arr_diag))):,}")
for s in INVALID_SENTINELS:
    print(f"  sentinel {s:>8.1f} count: {int(np.sum(arr_diag == s)):,}")
print(f"  Negative    : {int(np.sum(arr_diag < 0)):,}")
print(f"  > {MAX_MM_H:.0f} mm/h  : {int(np.sum(arr_diag > MAX_MM_H)):,}")
finite_arr = arr_diag[np.isfinite(arr_diag)]
if len(finite_arr) > 0:
    print(f"  Min (finite): {finite_arr.min():.4f}")
    print(f"  Max (finite): {finite_arr.max():.4f}")
    print(f"  Mean (all finite, no sentinel masking): {finite_arr.mean():.6f}")
mean_diag = load_mean(first_path)
print(f"  Mean rate (after full NaN handling): {mean_diag:.6f} mm/h")
print(f"  -> equivalent depth for this one 5-min window: "
      f"{mean_diag * HOURS_PER_FILE:.6f} mm")
print()

print(f"Loading {len(all_files):,} files with {N_WORKERS} workers …")
mean_buckets  = defaultdict(list)   
depth_buckets = defaultdict(float)  
skipped = 0

def process(item):
    sy, season, path = item
    return sy, season, load_mean(path)

with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
    futures = {ex.submit(process, item): item for item in all_files}
    done = 0
    for fut in as_completed(futures):
        sy, season, mean_rate = fut.result()
        if not np.isnan(mean_rate):
            mean_buckets[(sy, season)].append(mean_rate)
            depth_buckets[(sy, season)] += mean_rate * HOURS_PER_FILE
        else:
            skipped += 1
        done += 1
        if done % 50_000 == 0:
            pct = 100 * done / len(all_files)
            print(f"  … {done:,}/{len(all_files):,} ({pct:.1f}%) — skipped {skipped:,}")

print(f"\nDone. Skipped {skipped:,} files (all-NaN or unreadable).")

# ── BUILD TABLE ───────────────────────────────────────────────────────────────
seasons = ["Winter", "Spring", "Summer", "Autumn"]

print("\n--- Mean Intensity (mm/h) ---")
header = f"{'Year':<6}" + "".join(f"{s:>12}" for s in seasons)
print(header)
print("-" * len(header))

rows = []
for year in YEARS:
    row = {"Year": year}
    for season in seasons:
        vals = mean_buckets.get((year, season), [])
        row[f"{season}_mean"] = float(np.mean(vals)) if vals else float("nan")
        row[f"{season}_accum"] = depth_buckets.get((year, season), float("nan"))
    rows.append(row)
    cells = "".join(
        f"{row[f'{s}_mean']:>12.5f}" if not np.isnan(row[f"{s}_mean"]) else f"{'N/A':>12}"
        for s in seasons
    )
    print(f"{year:<6}{cells}")

print("\n--- Total Accumulation (mm) ---")
header2 = f"{'Year':<6}" + "".join(f"{s:>12}" for s in seasons) + f"{'Annual':>12}"
print(header2)
print("-" * len(header2))
for row in rows:
    annual = sum(row[f"{s}_accum"] for s in seasons if not np.isnan(row[f"{s}_accum"]))
    row["Annual_accum"] = annual
    cells = "".join(
        f"{row[f'{s}_accum']:>12.1f}" if not np.isnan(row[f"{s}_accum"]) else f"{'N/A':>12}"
        for s in seasons
    )
    print(f"{row['Year']:<6}{cells}{annual:>12.1f}")

out_csv_mean = Path("seasonal_rainfall_mean.csv")
with open(out_csv_mean, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Year"] + seasons)
    for row in rows:
        writer.writerow(
            [row["Year"]] + [
                f"{row[f'{s}_mean']:.5f}" if not np.isnan(row[f"{s}_mean"]) else "N/A"
                for s in seasons
            ]
        )
print(f"\nSaved mean intensities → {out_csv_mean.resolve()}")

out_csv_accum = Path("seasonal_rainfall_accumulation.csv")
with open(out_csv_accum, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Year"] + seasons + ["Annual"])
    for row in rows:
        writer.writerow(
            [row["Year"]] + [
                f"{row[f'{s}_accum']:.1f}" for s in seasons
            ] + [f"{row['Annual_accum']:.1f}"]
        )
print(f"Saved total accumulations → {out_csv_accum.resolve()}")