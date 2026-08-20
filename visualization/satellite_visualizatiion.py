import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime, timedelta

matplotlib.rcParams.update({
    "font.family":    "serif",
    "font.size":      14,
    "axes.linewidth": 1.5,
    "pdf.fonttype":   42,
    "ps.fonttype":    42,
    "svg.fonttype":   "none",
})

class SatGridConfig:

    out_dir   = 'SATELLITE_CH_GRID'
    out_fname = 'satellite_ch_grid.png'

    metadata_csv = 'metadata_patch/test_fullimage_summer.csv'
    num_samples  = 4   

    satellite_base = '/home/fe/sajib/scratch/weather-data/satellite_de'

    max_search_minutes = 90

    title = "SEVIRI RSS \u2013 Germany (CH7, CH9, CH9 \u2212 CH7)"


def _parse_time_token(fname):
    """'184500_CH7.npz' -> time(18,45,0)"""
    token = os.path.basename(fname).split('_')[0]
    return datetime.strptime(token, '%H%M%S').time()


def find_nearest_satellite_file(dt, satellite_base, channel, max_search_minutes):
    """
    Finds the CH7/CH9 file whose scan time is closest to `dt`, searching the
    day directory (and, near midnight, the adjacent day) since the RSS scan
    schedule is not a fixed 5-minute grid.
    """
    candidates = []
    for day_offset in (0, -1, 1):
        day = dt + timedelta(days=day_offset)
        day_dir = os.path.join(satellite_base, day.strftime('%Y'),
                                day.strftime('%m'), day.strftime('%d'))
        pattern = os.path.join(day_dir, f'*_{channel}.npz')
        for fpath in glob.glob(pattern):
            t = _parse_time_token(fpath)
            cand_dt = datetime.combine(day.date(), t)
            candidates.append((abs((cand_dt - dt).total_seconds()), cand_dt, fpath))

    if not candidates:
        raise FileNotFoundError(
            f"No {channel} satellite files found near {dt} under {satellite_base}"
        )

    candidates.sort(key=lambda c: c[0])
    best_delta_s, best_dt, best_path = candidates[0]
    if best_delta_s > max_search_minutes * 60:
        raise FileNotFoundError(
            f"Nearest {channel} scan to {dt} is {best_delta_s/60:.1f} min away "
            f"(> max_search_minutes={max_search_minutes}); found {best_path}"
        )
    return best_path, best_dt


def load_npz_array(path):
    """Loads the first array stored in an .npz file, regardless of its key name."""
    with np.load(path) as data:
        key = data.files[0]
        return data[key].astype(np.float32)


def load_ch7_ch9(dt, cfg):
    ch7_path, ch7_dt = find_nearest_satellite_file(dt, cfg.satellite_base, 'CH7', cfg.max_search_minutes)
    ch9_path, ch9_dt = find_nearest_satellite_file(dt, cfg.satellite_base, 'CH9', cfg.max_search_minutes)
    ch7 = load_npz_array(ch7_path)
    ch9 = load_npz_array(ch9_path)
    scan_dt = ch7_dt 
    return ch7, ch9, scan_dt

def _add_colorbar(fig, ax, im):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.08)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=12)
    return cbar


def save_satellite_grid(rows, cfg, out_path):
    """
    rows: list of dicts with keys 'ch7', 'ch9', 'dt' (representative scan time)
    """
    n_rows = len(rows)
    n_cols = 3
    col_titles = ["CH7", "CH9", "DIFF"]

    FIG_BG    = "white"
    TEXT_DARK = "#111111"

    IMG_W = 5.2
    ROW_H = 3.4

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(IMG_W * n_cols, ROW_H * n_rows),
                              facecolor=FIG_BG)
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(cfg.title, color=TEXT_DARK, fontsize=22, fontweight="bold", y=0.995)

    for r_idx, row in enumerate(rows):
        ch7 = row['ch7']
        ch9 = row['ch9']
        diff = ch9 - ch7

        diff_abs_max = float(np.nanmax(np.abs(diff))) if np.isfinite(diff).any() else 1.0

        panels = [
            (ch7,  dict(cmap="gray")),
            (ch9,  dict(cmap="gray")),
            (diff, dict(cmap="RdBu_r", vmin=-diff_abs_max, vmax=diff_abs_max)),
        ]

        for c_idx, (data, imshow_kwargs) in enumerate(panels):
            ax = axes[r_idx, c_idx]
            im = ax.imshow(data, interpolation="nearest", **imshow_kwargs)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)

            if r_idx == 0:
                ax.set_title(col_titles[c_idx], color=TEXT_DARK,
                             fontsize=17, fontweight="bold", pad=10)

            _add_colorbar(fig, ax, im)

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    pdf_path = os.path.splitext(out_path)[0] + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"  Saved: {out_path}")
    print(f"  Saved: {pdf_path}  (use this one in LaTeX via \\includegraphics)")


def main():
    cfg = SatGridConfig()

    print(f"Loading metadata: {cfg.metadata_csv}")
    df     = pd.read_csv(cfg.metadata_csv)
    df_top = df.nlargest(cfg.num_samples, 'p99(x_seq)').reset_index(drop=True)

    print(f"Top {cfg.num_samples} event timestamps (same selection as the radar GT grid):")
    print(df_top[['datetime', 'p99(x_seq)']].to_string(index=False))

    rows = []
    for idx, row in df_top.iterrows():
        dt = datetime.strptime(row['datetime'], '%Y-%m-%d %H:%M:%S')
        print(f"\n[{idx+1}/{cfg.num_samples}] target={dt} -> locating nearest CH7/CH9 scans ...")
        ch7, ch9, scan_dt = load_ch7_ch9(dt, cfg)
        print(f"  using scan time: {scan_dt}  (requested {dt})")
        rows.append({'ch7': ch7, 'ch9': ch9, 'dt': scan_dt})

    out_path = os.path.join(cfg.out_dir, cfg.out_fname)
    save_satellite_grid(rows, cfg, out_path)

    print(f"\nDone. Grid saved to: {out_path}")


if __name__ == "__main__":
    main()