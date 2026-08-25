import argparse
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count

BASE_DIR = Path("/home/fe/sajib/scratch/weather-data/radar_de")
YEARS    = list(range(2015, 2025))
SCALE    = 0.1   

THRESHOLDS = [0.1, 1.0, 5.0, 10.0]

def process_file(fp_str: str) -> tuple | None:
    """
    Load one .npy file and count pixels exceeding key thresholds.
    """
    try:
        arr = np.load(fp_str)
        arr = arr.ravel()
        valid = arr[(arr >= 0) & (~np.isnan(arr))]

        if valid.size == 0:
            return None

        if SCALE != 1.0:
            valid = valid * SCALE

        # Cumulative exceedance counts
        ge_0_1  = int(np.sum(valid >= 0.1))
        ge_1_0  = int(np.sum(valid >= 1.0))
        ge_5_0  = int(np.sum(valid >= 5.0))
        ge_10_0 = int(np.sum(valid >= 10.0))
        total   = int(valid.size)

        return (ge_0_1, ge_1_0, ge_5_0, ge_10_0, total)
    except Exception:
        return None


def collect_files(year: int) -> list[str]:
    year_str   = str(year)[2:]
    month_dirs = sorted(BASE_DIR.glob(f"{year_str}??"))
    if not month_dirs:
        raise FileNotFoundError(f"No folders matching {BASE_DIR}/{year_str}??")

    files = []
    for md in month_dirs:
        files.extend(str(p) for p in sorted(md.rglob("*.npy")))
    return files


def aggregate(raw: list) -> dict:
    ge_0_1 = ge_1_0 = ge_5_0 = ge_10_0 = total = 0
    for r in raw:
        if r is None:
            continue
        ge_0_1  += r[0]
        ge_1_0  += r[1]
        ge_5_0  += r[2]
        ge_10_0 += r[3]
        total   += r[4]

    return {
        "total": total,
        "p_0_1":  (100.0 * ge_0_1  / total) if total > 0 else 0.0,
        "p_1_0":  (100.0 * ge_1_0  / total) if total > 0 else 0.0,
        "p_5_0":  (100.0 * ge_5_0  / total) if total > 0 else 0.0,
        "p_10_0": (100.0 * ge_10_0 / total) if total > 0 else 0.0,
    }


def build_latex_table(rows: list[dict]) -> str:
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"    \centering")
    lines.append(r"    \caption{Annual precipitation threshold analysis.}")
    lines.append(r"    \label{tab:precip_threshold_analysis}")
    lines.append(r"    \begin{tabular}{l cccc}")
    lines.append(r"        \toprule")
    lines.append(
        r"        \textbf{Year} & \textbf{$\ge$0.1 mm/h (\%)} "
        r"& \textbf{$\ge$1.0 mm/h (\%)} & \textbf{$\ge$5.0 mm/h (\%)} "
        r"& \textbf{$\ge$10.0 mm/h (\%)} \\"
    )
    lines.append(r"        \midrule")

    for r in rows:
        lines.append(
            f"        {r['year']} & {r['p_0_1']:.3f} & {r['p_1_0']:.3f} "
            f"& {r['p_5_0']:.3f} & {r['p_10_0']:.3f} \\\\"
        )

    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"    \begin{tablenotes}")
    lines.append(r"        \small")
    lines.append(
        r"        \item Percentages computed from all valid 5-minute RADOLAN "
        r"composites for each year."
    )
    lines.append(r"    \end{tablenotes}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(cpu_count(), 16))
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir \
                 else BASE_DIR.parent / "radar_statistics" / "latex_tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for yr in YEARS:
        print(f"[{yr}] Processing...", end=" ", flush=True)
        try:
            files = collect_files(yr)
            with Pool(processes=args.workers) as pool:
                raw = pool.map(process_file, files, chunksize=256)

            result = aggregate(raw)
            rows.append({"year": yr, **result})
            print(f"Done. (≥0.1: {result['p_0_1']:.3f}%, ≥10.0: {result['p_10_0']:.3f}%)")

        except Exception as e:
            print(f"Error: {e}")

    latex = build_latex_table(rows)
    tex_path = output_dir / "table_3_6_precipitation_thresholds.tex"
    with open(tex_path, "w") as f:
        f.write(latex)

    print(f"\n--- LaTeX Table 3.6 Output ---\n")
    print(latex)
    print(f"\nSaved to: {tex_path}")


if __name__ == "__main__":
    main()