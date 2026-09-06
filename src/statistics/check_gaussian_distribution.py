import os
import random
import argparse
import numpy as np
from tqdm import tqdm
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sample_file_paths(satellite_root, channel, max_samples, seed):
    rng = random.Random(seed)
    years = sorted(os.listdir(satellite_root))
    picked = set()
    tries = 0
    max_tries = max_samples * 30

    while len(picked) < max_samples and tries < max_tries:
        tries += 1
        year = rng.choice(years)
        year_path = os.path.join(satellite_root, year)
        months = os.listdir(year_path)
        if not months:
            continue
        month = rng.choice(months)
        month_path = os.path.join(year_path, month)
        days = os.listdir(month_path)
        if not days:
            continue
        day = rng.choice(days)
        day_path = os.path.join(month_path, day)
        try:
            files = [f for f in os.listdir(day_path) if f.endswith(f"_CH{channel}.npy")]
        except NotADirectoryError:
            continue
        if not files:
            continue
        fname = rng.choice(files)
        picked.add(os.path.join(day_path, fname))

    return list(picked)


def load_pixels(paths, fill_value=None, nodata_value=None, positive_only=True):
    all_pixels = []
    n_failed = 0
    sample_errors = []
    for p in tqdm(paths, desc="  loading", ncols=100, leave=False):
        try:
            arr = np.load(p).astype(np.float32).ravel()
            arr = arr[np.isfinite(arr)]
            if fill_value is not None:
                arr = arr[arr != fill_value]
            if nodata_value is not None:
                arr = arr[arr != nodata_value]
            if positive_only:
                arr = arr[arr > 0.0]
            if len(arr) > 0:
                all_pixels.append(arr)
        except Exception as e:
            n_failed += 1
            if len(sample_errors) < 5:
                sample_errors.append(f"{p} | {type(e).__name__}: {e}")

    if n_failed:
        print(f"  note: {n_failed}/{len(paths)} files failed to load.")
        for e in sample_errors:
            print(f"    - {e}")

    if not all_pixels:
        raise RuntimeError("No valid pixels loaded — check --fill_value/--nodata_value/--satellite_root.")

    return np.concatenate(all_pixels)


def check_gaussian(name, pixels, out_dir, max_hist_samples=2_000_000, seed=42):
    rng = np.random.default_rng(seed)
    if len(pixels) > max_hist_samples:
        idx = rng.choice(len(pixels), max_hist_samples, replace=False)
        sample = pixels[idx]
    else:
        sample = pixels

    mean, std = sample.mean(), sample.std()
    skew = stats.skew(sample)
    kurt = stats.kurtosis(sample)

    sw_sample = sample if len(sample) <= 5000 else rng.choice(sample, 5000, replace=False)
    sw_stat, sw_p = stats.shapiro(sw_sample)

    print(f"\n{'='*60}")
    print(f"  {name}  (n={len(pixels):,}, plotted on n={len(sample):,})")
    print(f"{'='*60}")
    print(f"  mean               : {mean:.4f}")
    print(f"  std                : {std:.4f}")
    print(f"  skewness           : {skew:.4f}   (0 = perfectly symmetric)")
    print(f"  excess kurtosis    : {kurt:.4f}   (0 = normal-like tails)")
    print(f"  Shapiro-Wilk stat  : {sw_stat:.4f}, p-value: {sw_p:.4g}")
    print(f"  -> With large n, Shapiro-Wilk almost always rejects normality even for")
    print(f"     mild deviations. Judge mainly by the plots + |skew| and |kurtosis|")
    print(f"     (rule of thumb: |skew|<0.5 and |kurtosis|<1 = 'approximately Gaussian').")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].hist(sample, bins=80, density=True, color="#6FA8E8", edgecolor="none", alpha=0.85)
    x = np.linspace(sample.min(), sample.max(), 300)
    axes[0].plot(x, stats.norm.pdf(x, mean, std), color="#4A0B0B", linewidth=2, label="Fitted normal")
    axes[0].set_title(f"{name}: histogram vs fitted normal")
    axes[0].set_xlabel("Pixel value")
    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=False)

    stats.probplot(sample, dist="norm", plot=axes[1])
    axes[1].set_title(f"{name}: Q-Q plot")
    axes[1].get_lines()[0].set_markerfacecolor("#6FA8E8")
    axes[1].get_lines()[0].set_markeredgecolor("#0C447C")
    axes[1].get_lines()[1].set_color("#4A0B0B")

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{name.replace(' ', '_')}_gaussian_check.png")
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"  Saved plot -> {out_path}")

    return {"mean": mean, "std": std, "skew": skew, "kurtosis": kurt}


def main():
    parser = argparse.ArgumentParser(description="Check whether CH7/CH9 pixels are approximately Gaussian")
    parser.add_argument("--satellite_root",
                         default="/home/fe/sajib/scratch/weather-data/satellite_de_regridded")
    parser.add_argument("--channels", default="7,9", help="Comma-separated channel numbers")
    parser.add_argument("--max_samples", type=int, default=2000,
                         help="Number of .npy files to sample per channel (default: 2000)")
    parser.add_argument("--fill_value", type=float, default=None,
                         help="Fill/nodata sentinel value to drop, if any")
    parser.add_argument("--nodata_value", type=float, default=None)
    parser.add_argument("--no_positive_filter", action="store_true",
                         help="Don't drop values <= 0 (use if 0/negative are valid readings)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="gaussian_check")
    args = parser.parse_args()

    channels = [c.strip() for c in args.channels.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)

    results = {}
    for ch in channels:
        print(f"\nSampling files for Channel {ch} ...")
        paths = sample_file_paths(args.satellite_root, ch, args.max_samples, args.seed)
        print(f"  found {len(paths)} files")
        pixels = load_pixels(paths, fill_value=args.fill_value, nodata_value=args.nodata_value,
                              positive_only=not args.no_positive_filter)
        results[ch] = check_gaussian(f"Channel {ch}", pixels, args.out_dir, seed=args.seed)

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for ch, r in results.items():
        verdict = "approximately Gaussian" if abs(r["skew"]) < 0.5 and abs(r["kurtosis"]) < 1 else "NOT clearly Gaussian"
        print(f"  Channel {ch}: skew={r['skew']:.3f}, kurtosis={r['kurtosis']:.3f}  -> {verdict}")
    print()


if __name__ == "__main__":
    main()