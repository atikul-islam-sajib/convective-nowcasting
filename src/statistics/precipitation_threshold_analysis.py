import argparse
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
from collections import Counter

BASE_DIR = Path("/home/fe/sajib/scratch/weather-data/radar_de")
YEARS    = list(range(2015, 2025))
SCALE    = 1.0  
CLASSES  = ["No Rain", "Light", "Moderate", "Heavy"]
THRESHOLDS = [0.1, 1.0, 5.0, 10.0]


def process_file(fp_str: str):
    try:
        raw = np.load(fp_str, mmap_mode="r")
        dtype_str = str(raw.dtype)
        arr = raw.astype(np.float32) * SCALE
        arr = arr.ravel()
        valid = arr[(arr >= 0) & (~np.isnan(arr))]

        if valid.size == 0:
            return {"ok": False, "dtype": dtype_str, "error": "empty_after_filter"}

        no_rain  = int(np.sum(valid <  0.1))
        light    = int(np.sum((valid >= 0.1)  & (valid <  2.5)))
        moderate = int(np.sum((valid >= 2.5)  & (valid < 10.0)))
        heavy    = int(np.sum(valid >= 10.0))
        total    = int(valid.size)

        ge_0_1  = light + moderate + heavy         
        ge_1_0  = int(np.sum(valid >= 1.0))
        ge_5_0  = int(np.sum(valid >= 5.0))
        ge_10_0 = heavy                            

        return {
            "ok": True,
            "dtype": dtype_str,
            "no_rain": no_rain, "light": light, "moderate": moderate, "heavy": heavy,
            "ge_0_1": ge_0_1, "ge_1_0": ge_1_0, "ge_5_0": ge_5_0, "ge_10_0": ge_10_0,
            "total": total,
        }
    except Exception as e:
        return {"ok": False, "dtype": "UNKNOWN", "error": type(e).__name__}


def collect_files(year: int) -> list[str]:
    year_str   = str(year)[2:]
    month_dirs = sorted(BASE_DIR.glob(f"{year_str}??"))
    if not month_dirs:
        raise FileNotFoundError(
            f"No folders matching {BASE_DIR}/{year_str}?? "
            f"(expected e.g. {year_str}01 .. {year_str}12)"
        )
    files = []
    for md in month_dirs:
        files.extend(str(p) for p in sorted(md.rglob("*.npy")))
    return files


def aggregate(raw: list, n_files_found: int) -> dict:
    counts = {"No Rain": 0, "Light": 0, "Moderate": 0, "Heavy": 0}
    ge = {"ge_0_1": 0, "ge_1_0": 0, "ge_5_0": 0, "ge_10_0": 0}
    total = 0
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
        counts["No Rain"]  += r["no_rain"]
        counts["Light"]    += r["light"]
        counts["Moderate"] += r["moderate"]
        counts["Heavy"]    += r["heavy"]
        ge["ge_0_1"]  += r["ge_0_1"]
        ge["ge_1_0"]  += r["ge_1_0"]
        ge["ge_5_0"]  += r["ge_5_0"]
        ge["ge_10_0"] += r["ge_10_0"]
        total += r["total"]

    result = {
        "counts": counts,
        "total": total,
        "n_files_found": n_files_found,
        "n_ok": n_ok,
        "n_failed": n_failed,
        "error_counter": error_counter,
        "dtype_counter": dtype_counter,
    }
    for k, v in ge.items():
        result[k] = (100.0 * v / total) if total > 0 else 0.0
    return result


def build_table_3_6(rows: list[dict]) -> str:
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"    \centering")
    lines.append(r"    \caption{Annual precipitation threshold analysis.}")
    lines.append(r"    \label{tab:precip_threshold_analysis}")
    lines.append(r"    \begin{tabular}{l cccc}")
    lines.append(r"        \toprule")
    lines.append(
        r"        \textbf{Year} & \textbf{$\ge$0.1 mm/h (\%)} & \textbf{$\ge$1.0 mm/h (\%)} "
        r"& \textbf{$\ge$5.0 mm/h (\%)} & \textbf{$\ge$10.0 mm/h (\%)} \\"
    )
    lines.append(r"        \midrule")

    for r in rows:
        lines.append(
            f"        {r['year']} & {r['ge_0_1']:.3f} & {r['ge_1_0']:.3f} "
            f"& {r['ge_5_0']:.3f} & {r['ge_10_0']:.3f} \\\\"
        )

    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"    \begin{tablenotes}")
    lines.append(r"        \small")
    lines.append(r"        \item Percentages computed from all valid 5-minute RADOLAN composites for each year.")
    lines.append(r"        \item Computed from the same per-file pass used for Table~\ref{tab:radar_rainfall_dist} (Table 3.5), guaranteeing consistent totals across both tables.")
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
                 else BASE_DIR.parent / "radar_statistics" / "latex_tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    years_to_run = args.years if args.years else YEARS

    print(f"\n{'#'*60}")
    print(f"  Table 3.6 generation")
    print(f"  Years    : {years_to_run}")
    print(f"  Workers  : {args.workers}")
    print(f"  Output   : {output_dir}")
    print(f"{'#'*60}\n")

    rows = []
    for yr in years_to_run:
        print(f"[{yr}] Collecting files ...")
        try:
            files = collect_files(yr)
        except FileNotFoundError as e:
            print(f"  [ERROR] {e}")
            continue
        print(f"  {yr}: {len(files):,} files found on disk")

        with Pool(processes=args.workers) as pool:
            raw = pool.map(process_file, files, chunksize=256)

        result = aggregate(raw, n_files_found=len(files))
        rows.append({"year": yr, **result})

        t = result["total"]
        print(f"  Files OK / FAILED : {result['n_ok']:,} / {result['n_failed']:,}")
        if result["error_counter"]:
            print(f"  Failure reasons   : {dict(result['error_counter'])}")
        print(f"  Dtypes seen       : {dict(result['dtype_counter'])}")
        print(f"  TOTAL PIXELS      : {t:,}")
        print(f"  >=0.1mm/h (Table 3.6 style)   : {result['ge_0_1']:.3f}%")
        c = result["counts"]
        if t > 0:
            print(f"  Light+Mod+Heavy (Table 3.5 style): {100*(c['Light']+c['Moderate']+c['Heavy'])/t:.3f}%")
            print(f"  [these two lines must match -- if not, something is wrong in this script]\n")

    if not rows:
        print("No data processed. Check BASE_DIR and year folder names.")
        return

    tex_3_6 = build_table_3_6(rows)

    path_3_6 = output_dir / "table_3_6_final.tex"
    path_3_6.write_text(tex_3_6)

    print(f"\n--- Table 3.6 ---\n{tex_3_6}")
    print(f"\nSaved: {path_3_6}")


if __name__ == "__main__":
    main()