import argparse
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
from collections import defaultdict
from itertools import groupby

BASE_DIR       = Path("/home/fe/sajib/scratch/weather-data/radar_de")
YEARS          = list(range(2015, 2025))
SCALE          = 0.1        
RAIN_THRESHOLD = 0.1        

def process_file(fp_str: str) -> tuple | None:
    try:
        fp   = Path(fp_str)
        stem = fp.stem                      
        ymd  = stem.split("_")[0]            
        yr2  = int(ymd[:2])
        year = 2000 + yr2 if yr2 >= 15 else 2100 + yr2
        date_str = f"{year}{ymd[2:4]}{ymd[4:6]}"  
        arr = np.load(fp, mmap_mode="r").astype(np.float32) * SCALE
        arr = arr.ravel()
        arr = arr[arr >= 0]                    
        if arr.size == 0:
            return None

        mean      = float(np.mean(arr))
        maximum   = float(np.max(arr))
        std       = float(np.std(arr))
        rain_frac = float(np.mean(arr >= RAIN_THRESHOLD) * 100.0)

        return (date_str, mean, maximum, std, rain_frac)
    except Exception:
        return None


def collect_files(year: int) -> list[str]:
    year_str  = str(year)[2:]
    month_dirs = sorted(BASE_DIR.glob(f"{year_str}??"))
    if not month_dirs:
        raise FileNotFoundError(
            f"No folders found matching {BASE_DIR}/{year_str}?? "
            f"(expected e.g. {year_str}01 .. {year_str}12)"
        )
    files = []
    for md in month_dirs:
        files.extend(str(p) for p in sorted(md.rglob("*.npy")))
    print(f"  {year}: {len(month_dirs)} monthly dirs, {len(files):,} files")
    return files


def aggregate_yearly(results: list[tuple]) -> dict:
    daily = defaultdict(lambda: {"means": [], "maxs": [], "stds": [], "fracs": []})
    for date_str, mean, maximum, std, frac in results:
        daily[date_str]["means"].append(mean)
        daily[date_str]["maxs"].append(maximum)
        daily[date_str]["stds"].append(std)
        daily[date_str]["fracs"].append(frac)

    day_means, day_maxs, day_stds, day_fracs = [], [], [], []
    for d in sorted(daily):
        v = daily[d]
        day_means.append(np.mean(v["means"]))
        day_maxs.append(np.max(v["maxs"]))     
        day_stds.append(np.mean(v["stds"]))
        day_fracs.append(np.mean(v["fracs"]))

    return {
        "mean_rainfall": float(np.mean(day_means)),
        "max_rainfall":  float(np.mean(day_maxs)), 
        "std_dev":       float(np.mean(day_stds)),
        "rain_area":     float(np.mean(day_fracs)),
        "n_days":        len(day_means),
    }


def build_latex_table(rows: list[dict]) -> str:
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"    \centering")
    lines.append(
        r"    \caption{Annual RADOLAN Rainfall Statistics, Germany, 2015--2024}"
    )
    lines.append(r"    \label{tab:radar_daily_stats}")
    lines.append(r"    \begin{tabular}{l rrrr}")
    lines.append(r"        \toprule")
    lines.append(
        r"        \textbf{Year} "
        r"& \textbf{Mean (mm/h)} "
        r"& \textbf{Max (mm/h)} "
        r"& \textbf{Std Dev} "
        r"& \textbf{Rain Area (\%)} \\"
    )
    lines.append(r"        \midrule")

    for r in rows:
        lines.append(
            f"        {r['year']} "
            f"& {r['mean_rainfall']:>8.3f} "
            f"& {r['max_rainfall']:>8.3f} "
            f"& {r['std_dev']:>7.3f} "
            f"& {r['rain_area']:>8.1f} \\\\"
        )

    lines.append(r"        \midrule")
    lines.append(
        f"        \\textbf{{Mean}} "
        f"& \\textbf{{{np.mean([r['mean_rainfall'] for r in rows]):>8.3f}}} "
        f"& \\textbf{{{np.mean([r['max_rainfall']  for r in rows]):>8.3f}}} "
        f"& \\textbf{{{np.mean([r['std_dev']        for r in rows]):>7.3f}}} "
        f"& \\textbf{{{np.mean([r['rain_area']      for r in rows]):>8.1f}}} \\\\"
    )
    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(
        r"    \begin{tablenotes}\small"
        "\n"
        r"        \item Values are averages of daily statistics over all valid RADOLAN "
        r"5-minute composites. Rain Area = fraction of pixels $\geq$ 0.1\,mm/h."
        "\n"
        r"    \end{tablenotes}"
    )
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workers", type=int,
        default=min(cpu_count(), 16),
        help="Number of parallel workers (default: all cores up to 16)"
    )
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir \
                 else BASE_DIR.parent / "radar_statistics" / "latex_tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*56}")
    print(f"  Radar Daily Rainfall Statistics  2015–2024")
    print(f"  Workers  : {args.workers}")
    print(f"  Output   : {output_dir}")
    print(f"{'#'*56}\n")

    table_rows = []

    for yr in YEARS:
        print(f"\n[{yr}] Collecting files ...")
        try:
            files = collect_files(yr)
        except FileNotFoundError as e:
            print(f"  [ERROR] {e}")
            continue

        print(f"[{yr}] Processing with {args.workers} workers ...")
        with Pool(processes=args.workers) as pool:
            raw = pool.map(process_file, files, chunksize=256)

        results = [r for r in raw if r is not None]
        print(f"  Valid timesteps: {len(results):,} / {len(files):,}")

        stats = aggregate_yearly(results)
        table_rows.append({"year": yr, **stats})

        print(
            f"  Mean={stats['mean_rainfall']:.3f} mm/h  "
            f"Max={stats['max_rainfall']:.3f} mm/h  "
            f"Std={stats['std_dev']:.3f}  "
            f"RainArea={stats['rain_area']:.1f}%  "
            f"Days={stats['n_days']}"
        )

    if not table_rows:
        print("No data processed.")
        return

    latex = build_latex_table(table_rows)

    tex_path = output_dir / "radar_daily_stats_2015_2024.tex"
    with open(tex_path, "w") as f:
        f.write(latex)

    print(f"\n--- LaTeX Preview ---\n")
    print(latex)
    print(f"\n Saved: {tex_path}")


if __name__ == "__main__":
    main()