import argparse
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
from collections import Counter

BASE_DIR = Path("/home/fe/sajib/scratch/weather-data/radar_de")
YEARS    = list(range(2015, 2025))
SCALE    = 1.0
CLASSES  = ["No Rain", "Light", "Moderate", "Heavy"]


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
        light    = int(np.sum((valid >= 0.1) & (valid < 2.5)))
        moderate = int(np.sum((valid >= 2.5) & (valid < 10.0)))
        heavy    = int(np.sum(valid >= 10.0))
        total    = int(valid.size)

        assert no_rain + light + moderate + heavy == total, "class counts do not sum to total"

        return {
            "ok": True,
            "dtype": dtype_str,
            "no_rain": no_rain, "light": light, "moderate": moderate, "heavy": heavy,
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
        total += r["total"]

    return {
        "counts": counts,
        "total": total,
        "n_files_found": n_files_found,
        "n_ok": n_ok,
        "n_failed": n_failed,
        "error_counter": error_counter,
        "dtype_counter": dtype_counter,
    }


def build_table_3_5(rows: list[dict]) -> str:
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"    \centering")
    lines.append(r"    \caption{Annual RADOLAN Rainfall Intensity Distribution, Germany, 2015--2024}")
    lines.append(r"    \label{tab:radar_rainfall_dist}")
    lines.append(r"    \resizebox{\textwidth}{!}{%")
    lines.append(r"    \begin{tabular}{l rr rr rr rr}")
    lines.append(r"        \toprule")
    lines.append(
        r"        & \multicolumn{2}{c}{\textbf{No Rain ($<$0.1)}} "
        r"& \multicolumn{2}{c}{\textbf{Light (0.1--2.5)}} "
        r"& \multicolumn{2}{c}{\textbf{Moderate (2.5--10)}} "
        r"& \multicolumn{2}{c}{\textbf{Heavy ($>$10)}} \\"
    )
    lines.append(
        r"        \textbf{Year} & \textbf{Pixels} & \textbf{\%} & \textbf{Pixels} & \textbf{\%} "
        r"& \textbf{Pixels} & \textbf{\%} & \textbf{Pixels} & \textbf{\%} \\"
    )
    lines.append(r"        \midrule")

    for r in rows:
        t = r["total"]
        c = r["counts"]
        lines.append(
            f"        {r['year']} "
            f"& {c['No Rain']:>12,} & {100*c['No Rain']/t:>5.1f} "
            f"& {c['Light']:>10,} & {100*c['Light']/t:>5.1f} "
            f"& {c['Moderate']:>10,} & {100*c['Moderate']/t:>5.1f} "
            f"& {c['Heavy']:>8,} & {100*c['Heavy']/t:>5.1f} \\\\"
        )

    tot_pixels = sum(r["total"] for r in rows)
    tot_counts = {cls: sum(r["counts"][cls] for r in rows) for cls in CLASSES}
    lines.append(r"        \midrule")
    lines.append(
        f"        \\textbf{{Total}} "
        f"& \\textbf{{{tot_counts['No Rain']:>12,}}} & \\textbf{{{100*tot_counts['No Rain']/tot_pixels:>5.1f}}} "
        f"& \\textbf{{{tot_counts['Light']:>10,}}} & \\textbf{{{100*tot_counts['Light']/tot_pixels:>5.1f}}} "
        f"& \\textbf{{{tot_counts['Moderate']:>10,}}} & \\textbf{{{100*tot_counts['Moderate']/tot_pixels:>5.1f}}} "
        f"& \\textbf{{{tot_counts['Heavy']:>8,}}} & \\textbf{{{100*tot_counts['Heavy']/tot_pixels:>5.1f}}} \\\\"
    )
    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}}")
    lines.append(r"    \begin{tablenotes}")
    lines.append(r"        \small")
    lines.append(r"        \item Rainfall thresholds in mm/h. Pixel counts aggregated over all 5-minute RADOLAN composites per year.")
    lines.append(r"    \end{tablenotes}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(cpu_count(), 16))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--years", type=int, nargs="+", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir \
                 else BASE_DIR.parent / "radar_statistics" / "latex_tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    years_to_run = args.years if args.years else YEARS

    print(f"\n{'#'*60}")
    print(f"  Table 3.5 generation")
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
        print(f"  TOTAL PIXELS      : {t:,}\n")

    if not rows:
        print("No data processed. Check BASE_DIR and year folder names.")
        return

    tex_3_5 = build_table_3_5(rows)
    path_3_5 = output_dir / "table_3_5_final.tex"
    path_3_5.write_text(tex_3_5)

    print(f"\n--- Table 3.5 ---\n{tex_3_5}")
    print(f"\nSaved: {path_3_5}")


if __name__ == "__main__":
    main()