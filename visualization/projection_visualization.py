import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

matplotlib.rcParams.update({
    "font.family":    "serif",
    "font.size":      14,
    "axes.linewidth": 1.5,
    "pdf.fonttype":   42,
    "ps.fonttype":    42,
    "svg.fonttype":   "none",
})

class OverlayGridConfig:

    out_dir   = 'RADAR_SATELLITE_OVERLAY_GRID'
    out_fname = 'radar_satellite_overlay_grid.png'

    metadata_csv = 'metadata_patch/test_fullimage_summer.csv'
    num_samples  = 4 

    radar_base      = '/home/fe/sajib/scratch/weather-data/radar_de'
    satellite_base  = '/home/fe/sajib/scratch/weather-data/satellite_de_regridded'

    max_search_minutes = 90

    ch_low_pct, ch_high_pct       = 1.0, 99.0   
    radar_display_max_mmh         = 30.0        
    radar_rain_thr                = 0.1         

    title = "Radar\u2013Satellite (CH7 / CH9) Overlay Comparison"


def radar_path(base, dt):
    return os.path.join(base, dt.strftime('%y%m'), dt.strftime('%d'),
                        dt.strftime('%y%m%d_%H%M') + '.npy')


def load_radar_frame(dt, cfg):
    arr = np.load(radar_path(cfg.radar_base, dt)).astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 200.0)
    return arr


def _parse_time_token(fname):
    """'184500_CH7.npy' -> time(18,45,0)"""
    token = os.path.basename(fname).split('_')[0]
    return datetime.strptime(token, '%H%M%S').time()


def find_nearest_satellite_file(dt, satellite_base, channel, max_search_minutes):
    """
    Finds the CH7/CH9 file whose scan time is closest to `dt`, searching the
    day directory (and adjacent days near midnight), since the RSS scan
    schedule runs in irregular ~2h40m triplets rather than a fixed grid.
    """
    candidates = []
    for day_offset in (0, -1, 1):
        day = dt + timedelta(days=day_offset)
        day_dir = os.path.join(satellite_base, day.strftime('%Y'),
                                day.strftime('%m'), day.strftime('%d'))
        pattern = os.path.join(day_dir, f'*_{channel}.npy')
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


def load_ch7_ch9(dt, cfg):
    ch7_path, ch7_dt = find_nearest_satellite_file(dt, cfg.satellite_base, 'CH7', cfg.max_search_minutes)
    ch9_path, ch9_dt = find_nearest_satellite_file(dt, cfg.satellite_base, 'CH9', cfg.max_search_minutes)
    ch7 = np.nan_to_num(np.load(ch7_path).astype(np.float32), nan=0.0)
    ch9 = np.nan_to_num(np.load(ch9_path).astype(np.float32), nan=0.0)
    return ch7, ch9, ch7_dt  


def percentile_normalize(x, valid_mask, low_pct, high_pct):
    """Min-max stretch (by percentile) to [0, 1], restricted to valid pixels."""
    vals = x[valid_mask]
    if vals.size == 0:
        return np.zeros_like(x)
    lo, hi = np.percentile(vals, [low_pct, high_pct])
    if hi <= lo:
        hi = lo + 1e-6
    out = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    out[~valid_mask] = 0.0
    return out


def build_overlay(ch_norm, radar_norm, valid_mask):
    """R = channel (CH7 or CH9), G = 0, B = radar rain, masked to the
    satellite footprint so both channels go black outside it."""
    h, w = ch_norm.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[..., 0] = ch_norm * valid_mask
    rgb[..., 2] = radar_norm * valid_mask
    return rgb

def save_overlay_grid(rows, cfg, out_path):
    n_rows = len(rows)
    n_cols = 5
    TEXT_DARK = "#111111"

    IMG_W = 3.4
    ROW_H = 3.6

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(IMG_W * n_cols, ROW_H * n_rows),
                              facecolor="white")
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(cfg.title, color=TEXT_DARK, fontsize=20, fontweight="bold", y=0.995)

    for r_idx, row in enumerate(rows):
        dt        = row['dt']
        radar_mm  = row['radar']
        ch7       = row['ch7']
        ch9       = row['ch9']

        valid_mask = (ch7 != 0.0) | (ch9 != 0.0)

        ch7_norm   = percentile_normalize(ch7, valid_mask, cfg.ch_low_pct, cfg.ch_high_pct)
        ch9_norm   = percentile_normalize(ch9, valid_mask, cfg.ch_low_pct, cfg.ch_high_pct)
        radar_gate = radar_mm >= cfg.radar_rain_thr
        radar_norm = np.clip(radar_mm / cfg.radar_display_max_mmh, 0.0, 1.0) * radar_gate

        overlay_ch7 = build_overlay(ch7_norm, radar_norm, valid_mask)
        overlay_ch9 = build_overlay(ch9_norm, radar_norm, valid_mask)

        radar_display = np.where(radar_mm >= cfg.radar_rain_thr, radar_mm, np.nan)

        panels = [
            (radar_display, dict(cmap="Blues", vmin=0.0, vmax=cfg.radar_display_max_mmh), "radar"),
            (np.where(valid_mask, ch7_norm, np.nan), dict(cmap="gray", vmin=0.0, vmax=1.0), "gray"),
            (np.where(valid_mask, ch9_norm, np.nan), dict(cmap="gray", vmin=0.0, vmax=1.0), "gray"),
            (overlay_ch7, dict(), "rgb"),
            (overlay_ch9, dict(), "rgb"),
        ]

        col_titles = [
            f"{r_idx + 1}. {dt.strftime('%Y-%m-%d %H:%M:%S')}\nRadar",
            "CH7",
            "CH9",
            "Overlay: CH7 (R) + Radar (B)",
            "Overlay: CH9 (R) + Radar (B)",
        ]

        for c_idx, (data, imshow_kwargs, kind) in enumerate(panels):
            ax = axes[r_idx, c_idx]
            if kind == "radar":
                ax.set_facecolor("#f5f8fc")
            else:
                ax.set_facecolor("black")
            ax.imshow(data, interpolation="nearest", **imshow_kwargs)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_title(col_titles[c_idx], color=TEXT_DARK, fontsize=12,
                         fontweight="bold" if c_idx == 0 else "normal", pad=8)

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
    cfg = OverlayGridConfig()

    print(f"Loading metadata: {cfg.metadata_csv}")
    df     = pd.read_csv(cfg.metadata_csv)
    df_top = df.nlargest(cfg.num_samples, 'p99(x_seq)').reset_index(drop=True)

    print(f"Top {cfg.num_samples} event timestamps (same selection as the other grids):")
    print(df_top[['datetime', 'p99(x_seq)']].to_string(index=False))

    rows = []
    for idx, row in df_top.iterrows():
        dt = datetime.strptime(row['datetime'], '%Y-%m-%d %H:%M:%S')
        print(f"\n[{idx+1}/{cfg.num_samples}] target={dt}")

        radar_mm = load_radar_frame(dt, cfg)
        ch7, ch9, scan_dt = load_ch7_ch9(dt, cfg)
        print(f"  radar frame: {dt}   satellite scan: {scan_dt}")

        rows.append({'dt': dt, 'radar': radar_mm, 'ch7': ch7, 'ch9': ch9})

    out_path = os.path.join(cfg.out_dir, cfg.out_fname)
    save_overlay_grid(rows, cfg, out_path)

    print(f"\nDone. Grid saved to: {out_path}")


if __name__ == "__main__":
    main()