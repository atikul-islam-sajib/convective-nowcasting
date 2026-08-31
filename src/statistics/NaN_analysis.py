import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

RADAR_DIR    = Path("/home/fe/sajib/scratch/weather-data/radar_de")
YEARS        = list(range(2015, 2025))
SAMPLE_EVERY = 20 
N_WORKERS    = 8

def process_file(f):
    try:
        data = np.load(f, mmap_mode='r').ravel()
        total = len(data)
        nans  = int(np.sum(np.isnan(data) | (data < 0)))
        return total, nans
    except Exception:
        return 0, 0

for year in YEARS:
    year_short = str(year)[2:]
    files = []
    for month in range(1, 13):
        month_dir = RADAR_DIR / f"{year_short}{month:02d}"
        if not month_dir.exists():
            continue
        for day_dir in sorted(month_dir.iterdir()):
            if day_dir.is_dir():
                files.extend(sorted(day_dir.glob("*.npy")))

    files = files[::SAMPLE_EVERY]
    year_pixels = year_nans = 0

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        for total, nans in ex.map(process_file, files):
            year_pixels += total
            year_nans   += nans

    pct = 100.0 * year_nans / year_pixels if year_pixels > 0 else 0
    print(f"{year}: {year_nans:,} NaNs / {year_pixels:,} pixels ({pct:.2f}%)", flush=True)