import argparse
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count

BASE_DIR  = Path("/home/fe/sajib/scratch/weather-data/radar_de")
YEARS     = list(range(2015, 2025))
SCALE     = 0.1  
CLASSES   = ["No Rain", "Light", "Moderate", "Heavy"]

def process_file(fp_str: str) -> tuple | None:
    try:
        arr = np.load(fp_str, mmap_mode="r").astype(np.float32) * SCALE
        arr = arr.ravel()
        arr = arr[arr >= 0]          
        if arr.size == 0:
            return None

        no_rain  = int(np.sum(arr <  0.1))
        light    = int(np.sum((arr >= 0.1)  & (arr <  2.5)))
        moderate = int(np.sum((arr >= 2.5)  & (arr < 10.0)))
        heavy    = int(np.sum(arr >= 10.0))
        total    = int(arr.size)

        return (no_rain, light, moderate, heavy, total)
    except Exception:
        return None


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
    print(f"  {year}: {len(month_dirs)} monthly dirs, {len(files):,} files")
    return files


def aggregate(raw: list) -> dict:
    counts = {"No Rain": 0, "Light": 0, "Moderate": 0, "Heavy": 0}
    total  = 0
    for r in raw:
        if r is None:
            continue
        counts["No Rain"]  += r[0]
        counts["Light"]    += r[1]
        counts["Moderate"] += r[2]
        counts["Heavy"]    += r[3]
        total              += r[4]
    return {"counts": counts, "total": total}


def build_latex_table(rows: list[dict]) -> str:
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"    \centering")
    lines.append(
        r"    \caption{Annual RADOLAN Rainfall Intensity Distribution, "
        r"Germany, 2015--2024}"
    )
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
        r"        \textbf{Year} "
        r"& \textbf{Pixels} & \textbf{\%} "
        r"& \textbf{Pixels} & \textbf{\%} "
        r"& \textbf{Pixels} & \textbf{\%} "
        r"& \textbf{Pixels} & \textbf{\%} \\"
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
    lines.append(
        r"        \item Rainfall thresholds in mm/h. "
        r"Pixel counts aggregated over all 5-minute RADOLAN composites per year."
    )
    lines.append(r"    \end{tablenotes}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workers", type=int,
        default=min(cpu_count(), 16),
        help="Parallel workers (default: all cores up to 16)"
    )
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir \
                 else BASE_DIR.parent / "radar_statistics" / "latex_tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*56}")
    print(f"  Radar Rainfall Intensity Distribution  2015–2024")
    print(f"  Workers  : {args.workers}")
    print(f"  Output   : {output_dir}")
    print(f"{'#'*56}\n")

    rows = []
    for yr in YEARS:
        print(f"\n[{yr}] Collecting files ...")
        try:
            files = collect_files(yr)
        except FileNotFoundError as e:
            print(f"  [ERROR] {e}")
            continue

        print(f"[{yr}] Classifying pixels with {args.workers} workers ...")
        with Pool(processes=args.workers) as pool:
            raw = pool.map(process_file, files, chunksize=256)

        result = aggregate(raw)
        rows.append({"year": yr, **result})

        t = result["total"]
        c = result["counts"]
        print(f"  No Rain  : {c['No Rain']:>12,}  ({100*c['No Rain']/t:.1f}%)")
        print(f"  Light    : {c['Light']:>12,}  ({100*c['Light']/t:.1f}%)")
        print(f"  Moderate : {c['Moderate']:>12,}  ({100*c['Moderate']/t:.1f}%)")
        print(f"  Heavy    : {c['Heavy']:>12,}  ({100*c['Heavy']/t:.1f}%)")

    if not rows:
        print("No data processed. Check BASE_DIR and year folder names.")
        return

    latex = build_latex_table(rows)

    tex_path = output_dir / "radar_rainfall_distribution_2015_2024.tex"
    with open(tex_path, "w") as f:
        f.write(latex)

    print(f"\n--- LaTeX Preview ---\n")
    print(latex)
    print(f"\n Saved: {tex_path}")


if __name__ == "__main__":
    main()