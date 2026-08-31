import numpy as np
import pandas as pd
from pathlib import Path

RADAR_DIR   = Path("/home/fe/sajib/scratch/weather-data/radar_de")
OUTPUT_DIR  = Path("/home/fe/sajib/scratch/weather-data/radar_statistics")
OUTPUT_DIR.mkdir(exist_ok=True)

YEARS       = list(range(2015, 2024))
THRESHOLDS  = [0.1, 1.0, 5.0, 10.0]
PERCENTILES = [50, 90, 95, 99, 99.9]
CHUNK_SIZE  = 500  

def process_year(year):
    year_short = str(year)[2:]
    files = []

    for month in range(1, 13):
        month_dir = RADAR_DIR / f"{year_short}{month:02d}"
        if not month_dir.exists():
            continue
        for day_dir in sorted(month_dir.iterdir()):
            if day_dir.is_dir():
                files.extend(sorted(day_dir.glob("*.npy")))

    if not files:
        print(f"{year}: no files found")
        return year, None, None
    
    total_pixels   = 0
    exceed_counts  = {t: 0 for t in THRESHOLDS}

    nonzero_sample = []

    for i in range(0, len(files), CHUNK_SIZE):
        chunk_files = files[i:i + CHUNK_SIZE]
        chunk_data  = []

        for f in chunk_files:
            try:
                data = np.load(f, mmap_mode='r').astype(np.float32).ravel()
                data = data[(data >= 0) & (~np.isnan(data))]
                if data.size > 0:
                    chunk_data.append(data)
            except Exception as e:
                continue

        if not chunk_data:
            continue

        arr = np.concatenate(chunk_data)
        total_pixels += len(arr)

        for t in THRESHOLDS:
            exceed_counts[t] += int(np.sum(arr >= t))

        nz = arr[arr > 0]
        if nz.size > 0:
            sample_size = max(1, len(nz) // 100)
            idx = np.random.choice(len(nz), size=sample_size, replace=False)
            nonzero_sample.append(nz[idx])

        del arr, chunk_data
        print(f"  {year} chunk {i//CHUNK_SIZE + 1}/{(len(files)-1)//CHUNK_SIZE + 1} done", 
              flush=True)

    if total_pixels == 0:
        return year, None, None

    print(f"{year}: {total_pixels:,} total valid pixels", flush=True)

    exc = {"Year": year}
    for t in THRESHOLDS:
        exc[f">={t} mm/h (%)"] = round(100.0 * exceed_counts[t] / total_pixels, 4)


    sample = np.concatenate(nonzero_sample)
    pct = {"Year": year}
    for p in PERCENTILES:
        pct[f"p{p}"] = round(float(np.percentile(sample, p)), 4)

    return year, exc, pct

if __name__ == "__main__":
    np.random.seed(42)
    exceedance_rows = []
    percentile_rows = []

    for year in YEARS:
        y, exc, pct = process_year(year)
        if exc is not None:
            exceedance_rows.append(exc)
            percentile_rows.append(pct)

    df_exc = pd.DataFrame(exceedance_rows)
    df_pct = pd.DataFrame(percentile_rows)

    df_exc.to_csv(OUTPUT_DIR / "exceedance_probability.csv", index=False)
    df_pct.to_csv(OUTPUT_DIR / "percentile_distribution.csv", index=False)

    print("\n── Exceedance Probability ──")
    print(df_exc.to_string(index=False))

    print("\n── Percentile Distribution (non-zero pixels) ──")
    print(df_pct.to_string(index=False))