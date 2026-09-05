import argparse
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
from collections import Counter

RADAR_DIR = Path("/home/fe/sajib/scratch/weather-data/radar_de")
YEARS     = list(range(2015, 2025))


def process_file(fp_str: str):
    try:
        arr = np.load(fp_str, mmap_mode="r")
        dtype_str = str(arr.dtype)
        data = arr.ravel()
        total = int(data.size)
        nans  = int(np.sum(np.isnan(data) | (data < 0)))
        return {"ok": True, "dtype": dtype_str, "total": total, "nans": nans}
    except Exception as e:
        return {"ok": False, "dtype": "UNKNOWN", "error": type(e).__name__}


def collect_files(year: int) -> list[str]:
    year_short = str(year)[2:]
    files = []
    for month in range(1, 13):
        month_dir = RADAR_DIR / f"{year_short}{month:02d}"
        if not month_dir.exists():
            continue
        for day_dir in sorted(month_dir.iterdir()):
            if day_dir.is_dir():
                files.extend(str(p) for p in sorted(day_dir.glob("*.npy")))
    if not files:
        raise FileNotFoundError(
            f"No files found under {RADAR_DIR}/{year_short}?? -- check folder naming."
        )
    return files


def aggregate(raw: list, n_files_found: int) -> dict:
    total = nans = 0
    n_ok = n_failed = 0
    error_counter = Counter()
    dtype_counter = Counter()

    for r in raw:
        dtype_counter[r["dtype"]] += 1
        if not r["ok"]:
            n_failed += 1
            error_counter[r["error"]] += 1
            continue
        n_ok += 1
        total += r["total"]
        nans  += r["nans"]

    pct = 100.0 * nans / total if total > 0 else 0.0
    return {
        "total_pixels": total,
        "nan_pixels": nans,
        "nan_pct": pct,
        "n_files_found": n_files_found,
        "n_ok": n_ok,
        "n_failed": n_failed,
        "error_counter": error_counter,
        "dtype_counter": dtype_counter,
    }


def build_latex_table(rows: list[dict]) -> str:
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"    \centering")
    lines.append(r"    \caption{Year-wise distribution of NaN values in the RADOLAN radar dataset.}")
    lines.append(r"    \label{tab:radar_nan}")
    lines.append(r"    \begin{tabular}{l r r r}")
    lines.append(r"        \toprule")
    lines.append(
        r"        \textbf{Year} & \textbf{Total Pixels} & \textbf{NaN Pixels} & \textbf{NaN (\%)} \\"
    )
    lines.append(r"        \midrule")

    for r in rows:
        lines.append(
            f"        {r['year']} & {r['total_pixels']:,} & {r['nan_pixels']:,} & {r['nan_pct']:.2f} \\\\"
        )

    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"    \begin{tablenotes}")
    lines.append(r"        \small")
    lines.append(
        r"        \item The NaN percentage is calculated as the ratio of NaN pixels to the total "
        r"number of radar pixels for each year, computed over the full annual file set "
        r"(no subsampling), consistent with Tables~\ref{tab:radar_rainfall_dist} and \ref{tab:exceedance}."
    )
    lines.append(r"    \end{tablenotes}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(cpu_count(), 16))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--years", type=int, nargs="+", default=None,
                         help="Restrict to specific years, e.g. --years 2024")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir \
                 else RADAR_DIR.parent / "radar_statistics" / "latex_tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    years_to_run = args.years if args.years else YEARS

    print(f"\n{'#'*60}")
    print(f"  Table 3.8 -- Radar NaN Analysis (FULL SCAN, no subsampling)")
    print(f"  Years    : {years_to_run}")
    print(f"  Workers  : {args.workers}")
    print(f"  Output   : {output_dir}")
    print(f"{'#'*60}\n")

    rows = []
    for year in years_to_run:
        print(f"[{year}] Collecting files ...")
        try:
            files = collect_files(year)
        except FileNotFoundError as e:
            print(f"  [ERROR] {e}")
            continue
        print(f"  {year}: {len(files):,} files found on disk")

        with Pool(processes=args.workers) as pool:
            raw = pool.map(process_file, files, chunksize=256)

        result = aggregate(raw, n_files_found=len(files))
        rows.append({"year": year, **result})

        print(f"  Files OK / FAILED : {result['n_ok']:,} / {result['n_failed']:,}")
        if result["error_counter"]:
            print(f"  Failure reasons   : {dict(result['error_counter'])}")
        print(f"  Dtypes seen       : {dict(result['dtype_counter'])}")
        print(f"  TOTAL PIXELS      : {result['total_pixels']:,}")
        print(f"  NaN PIXELS        : {result['nan_pixels']:,}")
        print(f"  NaN %             : {result['nan_pct']:.2f}%\n")

    if not rows:
        print("No data processed. Check RADAR_DIR and year folder names.")
        return

    latex = build_latex_table(rows)
    tex_path = output_dir / "table_3_8_final.tex"
    tex_path.write_text(latex)

    print(f"\n--- Table 3.8 ---\n{latex}")
    print(f"\nSaved: {tex_path}")


if __name__ == "__main__":
    main()