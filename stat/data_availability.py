import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm
from collections import defaultdict

plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize":11,
    "figure.dpi":     150,
})

YEARS      = list(range(2015, 2025))
EXPECTED   = 288                      
LEAP_YEARS = {2016, 2020, 2024}
PALETTE    = ["#1E4682", "#3A7FBF", "#E87722", "#D94F3D"]


def days_in_year(y):
    return 366 if y in LEAP_YEARS else 365


def radar_files_for_year(radar_root, year):
    yy = str(year)[2:]                 
    pattern = os.path.join(radar_root, f"{yy}??", "??", "*.npy")
    files = glob.glob(pattern)
    return files


def sat_files_for_year(sat_root, year, channel="CH9"):
    pattern = os.path.join(sat_root, str(year), "??", "??", f"*_{channel}.npz")
    files = glob.glob(pattern)
    return files


def load_radar_sample(path):
    return np.load(path)


def load_sat_sample(path):
    d = np.load(path, allow_pickle=True)
    return d["data"].astype(np.float32)


def save(fig, output_dir, name):
    os.makedirs(output_dir, exist_ok=True)
    p = os.path.join(output_dir, name)
    fig.savefig(p, bbox_inches="tight")
    print(f"  saved → {p}")
    plt.close(fig)

def compute_availability(radar_root, sat_root):
    radar_counts, sat_counts = {}, {}
    for y in YEARS:
        rf = radar_files_for_year(radar_root, y)
        sf = sat_files_for_year(sat_root, y, "CH9")
        radar_counts[y] = len(rf)
        sat_counts[y]   = len(sf)
        print(f"    {y}: radar={len(rf):,}  satellite={len(sf):,}")
    return radar_counts, sat_counts


def plot_availability_table(radar_counts, sat_counts, output_dir):
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.axis("off")

    cols = ["Year", "Expected\nTimesteps",
            "Radar\nAvailable", "Radar\nMissing (%)",
            "Satellite\nAvailable", "Satellite\nMissing (%)"]
    rows = []
    for y in YEARS:
        exp   = days_in_year(y) * EXPECTED
        ra    = radar_counts[y]
        sa    = sat_counts[y]
        r_mis = f"{max(0, exp-ra)/exp*100:.1f}%"
        s_mis = f"{max(0, exp-sa)/exp*100:.1f}%"
        rows.append([str(y), f"{exp:,}", f"{ra:,}", r_mis, f"{sa:,}", s_mis])

    tbl = ax.table(cellText=rows, colLabels=cols,
                   loc="center", cellLoc="center",
                   bbox=[0, 0, 1, 0.88])          

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10.5)

    for j in range(len(cols)):
        cell = tbl[0, j]
        cell.set_facecolor("#1E4682")
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_height(0.13)

    for i in range(1, len(rows) + 1):
        fc = "#EEF2F8" if i % 2 == 0 else "white"
        for j in range(len(cols)):
            tbl[i, j].set_facecolor(fc)
            tbl[i, j].set_height(0.085)

    ax.set_title("Data Availability Summary (2015–2024)",
                 fontsize=14, fontweight="bold",
                 loc="center", pad=6,
                 y=0.98)

    fig.tight_layout()
    save(fig, output_dir, "stat1_availability_table.png")


def plot_missing_per_year(radar_counts, sat_counts, output_dir):
    exps   = [days_in_year(y) * EXPECTED for y in YEARS]
    r_miss = [max(0, e - radar_counts[y]) for y, e in zip(YEARS, exps)]
    s_miss = [max(0, e - sat_counts[y])   for y, e in zip(YEARS, exps)]

    x = np.arange(len(YEARS))
    w = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w/2, r_miss, w, label="Radar",     color=PALETTE[0], alpha=0.88)
    ax.bar(x + w/2, s_miss, w, label="Satellite", color=PALETTE[2], alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(YEARS)
    ax.set_xlabel("Year")
    ax.set_ylabel("Missing Timesteps")
    ax.set_title("Missing Timesteps per Year – Radar vs. Satellite")
    ax.legend()
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    save(fig, output_dir, "stat2_missing_per_year.png")


def plot_bt_distribution(sat_root, output_dir, n_sample=500):
    print(f"  Sampling up to {n_sample} files per channel …")

    ch7_files = glob.glob(os.path.join(sat_root, "????", "??", "??", "*_CH7.npz"))
    ch9_files = glob.glob(os.path.join(sat_root, "????", "??", "??", "*_CH9.npz"))

    print(f"  Found: CH7={len(ch7_files):,}  CH9={len(ch9_files):,}")

    if len(ch7_files) == 0 or len(ch9_files) == 0:
        print("  WARNING: No satellite files found. Check --sat_root path.")
        return None, None

    ch7_sample = np.random.choice(ch7_files, min(n_sample, len(ch7_files)), replace=False)
    ch9_sample = np.random.choice(ch9_files, min(n_sample, len(ch9_files)), replace=False)

    ch7_vals, ch9_vals = [], []
    for f in ch7_sample:
        try:
            arr = load_sat_sample(f).ravel()
            arr = arr[np.isfinite(arr)]
            if len(arr) > 0:
                ch7_vals.append(arr)
        except Exception as e:
            pass

    for f in ch9_sample:
        try:
            arr = load_sat_sample(f).ravel()
            arr = arr[np.isfinite(arr)]
            if len(arr) > 0:
                ch9_vals.append(arr)
        except Exception as e:
            pass

    print(f"  Loaded: CH7={len(ch7_vals)} arrays  CH9={len(ch9_vals)} arrays")

    if len(ch7_vals) == 0 or len(ch9_vals) == 0:
        print("  WARNING: Could not load satellite data arrays.")
        return None, None

    ch7_vals = np.concatenate(ch7_vals)
    ch9_vals = np.concatenate(ch9_vals)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(ch7_vals, bins=120, density=True,
            alpha=0.65, color=PALETTE[0], label="CH7 (IR 8.7 µm)")
    ax.hist(ch9_vals, bins=120, density=True,
            alpha=0.65, color=PALETTE[2], label="CH9 (IR 10.8 µm)")
    ax.set_xlabel("Brightness Temperature (K)")
    ax.set_ylabel("Density")
    ax.set_title("Brightness Temperature Distribution – CH7 & CH9")
    ax.legend()
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    save(fig, output_dir, "stat3_bt_distribution.png")
    return ch7_vals, ch9_vals


def plot_diff_distribution(sat_root, output_dir, n_sample=300):
    print(f"  Computing CH9-CH7 difference from {n_sample} paired files …")

    ch7_files = sorted(glob.glob(
        os.path.join(sat_root, "????", "??", "??", "*_CH7.npz")))

    if len(ch7_files) == 0:
        print("  WARNING: No CH7 files found.")
        return

    chosen = np.random.choice(ch7_files, min(n_sample, len(ch7_files)), replace=False)
    diff_vals = []

    for f7 in chosen:
        f9 = f7.replace("_CH7.npz", "_CH9.npz")
        if not os.path.exists(f9):
            continue
        try:
            d7 = load_sat_sample(f7).ravel()
            d9 = load_sat_sample(f9).ravel()
            diff = d9 - d7
            diff = diff[np.isfinite(diff)]
            if len(diff) > 0:
                diff_vals.append(diff)
        except Exception:
            pass

    if len(diff_vals) == 0:
        print("  WARNING: No valid paired files found.")
        return

    diff_vals = np.concatenate(diff_vals)
    print(f"  Difference values: min={diff_vals.min():.2f}  max={diff_vals.max():.2f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(diff_vals, bins=150, density=True,
            color=PALETTE[3], alpha=0.80, label="CH9 − CH7")
    ax.axvline(0, color="black", lw=1.4, linestyle="--", label="Zero line")
    ax.set_xlabel("Brightness Temperature Difference (CH9 − CH7)  [K]")
    ax.set_ylabel("Density")
    ax.set_title("CH9 − CH7 Difference Distribution\n"
                 "Negative values indicate deep convective clouds")
    ax.legend()
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    save(fig, output_dir, "stat4_diff_distribution.png")


def plot_spatial_mean(radar_root, output_dir, n_sample=2000):
    print(f"  Loading up to {n_sample} radar files for spatial mean …")
    files = glob.glob(os.path.join(radar_root, "????", "??", "*.npy"))
    print(f"  Found {len(files):,} radar files")

    if len(files) == 0:
        print("  WARNING: No radar files found. Check --radar_root path.")
        return

    chosen = np.random.choice(files, min(n_sample, len(files)), replace=False)
    acc   = None
    count = 0

    for f in chosen:
        try:
            arr = load_radar_sample(f).astype(np.float32)
            arr[arr < 0] = np.nan
            if acc is None:
                acc = np.zeros(arr.shape, dtype=np.float64)
            if arr.shape == acc.shape:
                valid = np.where(np.isfinite(arr), arr, 0.0)
                acc  += valid
                count += 1
        except Exception:
            pass

    if count == 0 or acc is None:
        print("  WARNING: Could not load any radar arrays.")
        return

    mean_map = acc / count
    pos_vals = mean_map[mean_map > 0]
    vmax     = float(np.percentile(pos_vals, 99)) if len(pos_vals) > 0 else 1.0

    fig, ax = plt.subplots(figsize=(7, 8))
    im = ax.imshow(mean_map, origin="upper",
                   norm=LogNorm(vmin=0.001, vmax=max(vmax, 0.01)),
                   cmap="Blues")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cb.set_label("Mean Precipitation (mm / 5 min, log scale)")
    ax.set_title(f"10-Year Spatial Mean Precipitation\nRADOLAN Domain (2015–2024)\n"
                 f"(averaged over {count:,} random timesteps)")
    ax.axis("off")
    fig.tight_layout()
    save(fig, output_dir, "stat5_spatial_mean_precip.png")


def plot_diurnal_cycle(radar_root, output_dir, n_sample=50000):
    print("  Computing diurnal cycle …")
    files = glob.glob(os.path.join(radar_root, "????", "??", "*.npy"))
    print(f"  Found {len(files):,} radar files")

    if len(files) == 0:
        print("  WARNING: No radar files found.")
        return

    if len(files) > n_sample:
        files = list(np.random.choice(files, n_sample, replace=False))

    hourly_sum   = defaultdict(float)
    hourly_count = defaultdict(int)

    for f in files:
        basename = os.path.basename(f)        
        try:
            time_part = basename.split("_")[1].replace(".npy", "")  
            hour = int(time_part[:2])
            arr  = load_radar_sample(f).astype(np.float32)
            arr[arr < 0] = np.nan
            m = float(np.nanmean(arr))
            if np.isfinite(m):
                hourly_sum[hour]   += m
                hourly_count[hour] += 1
        except Exception:
            pass

    hours = sorted(hourly_sum.keys())
    means = [hourly_sum[h] / hourly_count[h] if hourly_count[h] > 0 else 0
             for h in hours]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(hours, means, color=PALETTE[0], lw=2.5,
            marker="o", markersize=5, label="Mean precipitation")
    ax.fill_between(hours, means, alpha=0.18, color=PALETTE[0])
    ax.set_xticks(range(0, 24))
    ax.set_xlabel("Hour of Day (UTC)")
    ax.set_ylabel("Mean Precipitation (mm / 5 min)")
    ax.set_title("Diurnal Cycle of Precipitation – RADOLAN (2015–2024)")
    ax.legend()
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    save(fig, output_dir, "stat6_diurnal_cycle.png")

def plot_spatial_percentiles(radar_root, output_dir, n_sample=3000):
    print(f"  Loading up to {n_sample} radar files for percentile maps …")
    files = glob.glob(os.path.join(radar_root, "????", "??", "*.npy"))
    print(f"  Found {len(files):,} radar files")

    if len(files) == 0:
        print("  WARNING: No radar files found.")
        return

    chosen = np.random.choice(files, min(n_sample, len(files)), replace=False)

    stack = []
    shape = None
    for f in chosen:
        try:
            arr = load_radar_sample(f).astype(np.float32)
            arr[arr < 0] = np.nan
            if shape is None:
                shape = arr.shape
            if arr.shape == shape:
                stack.append(arr)
        except Exception:
            pass

    if len(stack) == 0:
        print("  WARNING: Could not load any radar arrays.")
        return

    stack = np.stack(stack, axis=0)        
    print(f"  Computing percentiles over {stack.shape[0]:,} timesteps …")

    from matplotlib.colors import LinearSegmentedColormap
    radar_colors = [
        (0.40, 0.40, 0.40),   # grey  – no/very low rain
        (0.00, 0.60, 0.00),   # green – light rain
        (1.00, 1.00, 0.00),   # yellow
        (1.00, 0.55, 0.00),   # orange
        (1.00, 0.10, 0.10),   # red
        (0.85, 0.00, 0.85),   # magenta – extreme
    ]
    radar_cmap = LinearSegmentedColormap.from_list(
        "radar", radar_colors, N=256)
    radar_cmap.set_bad(color="#2a2a2a")      

    percentiles = [90, 95, 99, 99.9]
    maps        = [np.nanpercentile(stack, p, axis=0) for p in percentiles]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.ravel()

    for i, (p, pmap) in enumerate(zip(percentiles, maps)):
        ax    = axes[i]
        pmap  = np.ma.masked_where(~np.isfinite(pmap) | (pmap <= 0), pmap)
        pos   = pmap.compressed()
        if len(pos) == 0:
            ax.set_title(f"P{p}")
            continue
        vmin  = max(0.001, float(np.nanpercentile(pos, 1)))
        vmax  = float(np.nanpercentile(pos, 99))
        im    = ax.imshow(pmap, origin="upper",
                          norm=LogNorm(vmin=max(vmin, 1e-4),
                                      vmax=max(vmax, vmin * 2)),
                          cmap=radar_cmap)
        cb    = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("mm / 5 min", fontsize=10)
        ax.set_title(f"P{p} Precipitation Percentile",
                     fontsize=13, fontweight="bold")
        ax.axis("off")

    fig.suptitle(
        "Spatial Precipitation Percentile Maps – RADOLAN (2015–2024)\n"
        f"Computed over {len(stack):,} randomly sampled timesteps",
        fontsize=14, fontweight="bold"
    )
    fig.tight_layout()
    save(fig, output_dir, "stat7_spatial_percentile_maps.png")


def plot_exceedance_curve(radar_root, output_dir, n_sample=5000):
    """
    For a set of thresholds, compute the fraction of all pixels × timesteps
    that exceed the threshold. Produces the classic exceedance probability
    curve — directly motivates the convective event threshold choice.
    """
    print(f"  Computing exceedance probability curve …")
    files = glob.glob(os.path.join(radar_root, "????", "??", "*.npy"))
    print(f"  Found {len(files):,} radar files")

    if len(files) == 0:
        print("  WARNING: No radar files found.")
        return

    chosen = np.random.choice(files, min(n_sample, len(files)), replace=False)

    all_vals = []
    for f in chosen:
        try:
            arr = load_radar_sample(f).astype(np.float32)
            arr = arr[arr >= 0]             
            arr = arr[np.isfinite(arr)]
            if len(arr) > 0:
                idx = np.random.choice(len(arr), min(500, len(arr)),
                                       replace=False)
                all_vals.append(arr[idx])
        except Exception:
            pass

    if len(all_vals) == 0:
        print("  WARNING: No valid radar values found.")
        return

    all_vals = np.concatenate(all_vals)
    print(f"  Total values for exceedance: {len(all_vals):,}")

    thresholds = np.logspace(-3, 2, 300)    
    exceed_prob = [np.mean(all_vals > t) for t in thresholds]

    ref_thresholds = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    ref_probs      = [np.mean(all_vals > t) for t in ref_thresholds]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(thresholds, exceed_prob,
            color=PALETTE[0], lw=2.5, label="Exceedance probability")

    for t, prob in zip(ref_thresholds, ref_probs):
        if prob > 0:
            ax.axvline(t, color="grey", lw=0.8, linestyle="--", alpha=0.6)
            ax.annotate(f"{t} mm\n({prob*100:.2f}%)",
                        xy=(t, prob),
                        xytext=(t * 1.6, prob * 1.8),
                        fontsize=8.5,
                        arrowprops=dict(arrowstyle="->", color="grey",
                                        lw=0.8),
                        color="#333333")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Precipitation Threshold (mm / 5 min)")
    ax.set_ylabel("Fraction of Pixels Exceeding Threshold")
    ax.set_title(
        "Precipitation Exceedance Probability Curve – RADOLAN (2015–2024)\n"
        "Illustrates class imbalance and motivates convective event threshold"
    )
    ax.grid(which="both", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    save(fig, output_dir, "stat8_exceedance_curve.png")


def main():
    parser = argparse.ArgumentParser(description="Dataset statistics & plots")
    parser.add_argument("--radar_root",   required=True)
    parser.add_argument("--sat_root",     required=True)
    parser.add_argument("--output_dir",   default="./figures")
    parser.add_argument("--sat_sample",   type=int, default=500)
    parser.add_argument("--radar_sample", type=int, default=2000)
    args = parser.parse_args()
    
    for label, path in [("--radar_root", args.radar_root),
                        ("--sat_root",   args.sat_root)]:
        if not os.path.isdir(path):
            print(f"ERROR: {label} path does not exist: {path}")
            return

    print("=" * 60)
    print("  Dataset Statistics")
    print("=" * 60)
    print(f"  radar_root : {args.radar_root}")
    print(f"  sat_root   : {args.sat_root}")
    print(f"  output_dir : {args.output_dir}")

    print("\n[1/6] Computing data availability …")
    radar_counts, sat_counts = compute_availability(
        args.radar_root, args.sat_root)
    plot_availability_table(radar_counts, sat_counts, args.output_dir)

    print("\n[2/6] Plotting missing data per year …")
    plot_missing_per_year(radar_counts, sat_counts, args.output_dir)

    print("\n[3/6] Brightness temperature distribution …")
    plot_bt_distribution(args.sat_root, args.output_dir,
                         n_sample=args.sat_sample)

    print("\n[4/6] CH9 - CH7 difference distribution …")
    plot_diff_distribution(args.sat_root, args.output_dir,
                           n_sample=args.sat_sample)

    print("\n[5/6] Spatial mean precipitation map …")
    plot_spatial_mean(args.radar_root, args.output_dir,
                      n_sample=args.radar_sample)

    print("\n[6/6] Diurnal cycle of precipitation …")
    plot_diurnal_cycle(args.radar_root, args.output_dir)

    print("\n[7/8] Spatial percentile maps …")
    plot_spatial_percentiles(args.radar_root, args.output_dir,
                             n_sample=args.radar_sample)

    print("\n[8/8] Exceedance probability curve …")
    plot_exceedance_curve(args.radar_root, args.output_dir,
                          n_sample=5000)

    print("\n" + "=" * 60)
    print(f"  All figures saved to: {args.output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()