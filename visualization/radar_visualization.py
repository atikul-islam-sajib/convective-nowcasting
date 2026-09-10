import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from datetime import datetime, timedelta

from skimage.morphology import closing, remove_small_objects, disk
from scipy.ndimage import label

matplotlib.rcParams.update({
    "font.family":      "serif",        
    "font.size":        14,
    "axes.linewidth":   1.5,
    "pdf.fonttype":     42,              
    "ps.fonttype":      42,
    "svg.fonttype":     "none",
})


CLIP_MAX_MMH = 128.0

RAIN_LEVELS = [0.1, 1, 2, 5, 10, 15, 20, 30, 40, 60, 100]
RAIN_COLORS = [
    "#1a5c1a", "#22aa22", "#55cc22", "#ffee00", "#ffaa00",
    "#ff6600", "#ff2200", "#cc0000", "#aa0077", "#ff00ff",
]
RAIN_CMAP = ListedColormap(RAIN_COLORS)
RAIN_NORM = BoundaryNorm(RAIN_LEVELS, RAIN_CMAP.N)
BG_COLOR  = "#888888"


class GTGridConfig:

    out_dir      = 'RADAR_GT_GRID'
    out_fname    = 'radar_gt_grid.png'
    metadata_csv = 'metadata_patch/test_fullimage_summer.csv'

    radar_base = '/home/fe/sajib/scratch/weather-data/radar_de'

    num_samples = 4                       
    horizons    = [15, 30, 45, 60]         

    operational_thr      = 15.0
    extreme_thr          = 40.0
    storm_min_pixels     = 200
    morphology_disk_size = 4
    storm_min_area_km2   = 200.0
    storm_max_area_km2   = 10000.0
    pixel_area_km2       = 1.0
    show_storm_metrics   = False  

    title = "Radar Snapshots of Convective Storm Events"


def radar_path(base, dt):
    return os.path.join(base, dt.strftime('%y%m'), dt.strftime('%d'),
                        dt.strftime('%y%m%d_%H%M') + '.npy')


def get_target_times(current_time, horizons_minutes):
    return [current_time + timedelta(minutes=h) for h in horizons_minutes]


def load_ground_truth_sample(row, cfg):
    dt           = datetime.strptime(row['datetime'], '%Y-%m-%d %H:%M:%S')
    target_times = get_target_times(dt, cfg.horizons)

    targets_list, masks_list = [], []
    for t in target_times:
        arr  = np.load(radar_path(cfg.radar_base, t)).astype(np.float32)
        mask = np.isfinite(arr).astype(np.float32)
        arr  = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr  = np.clip(arr, 0.0, CLIP_MAX_MMH)
        targets_list.append(arr)
        masks_list.append(mask)

    return np.stack(targets_list), np.stack(masks_list), dt


def detect_storms_two_level(rain_mm_h, operational_thr, extreme_thr,
                             min_pixels=10, disk_size=4,
                             min_area_km2=10.0, max_area_km2=100000.0,
                             pixel_area_km2=1.0):
    valid = np.isfinite(rain_mm_h)
    if valid.sum() == 0:
        empty = np.zeros_like(rain_mm_h, dtype=bool)
        return empty, empty, 0, 0
    mask_op = valid & (rain_mm_h >= operational_thr)
    if mask_op.sum() == 0:
        empty = np.zeros_like(rain_mm_h, dtype=bool)
        return empty, empty, 0, 0
    mask_op         = closing(mask_op, disk(disk_size))
    labeled_op, n_c = label(mask_op)
    final_op        = np.zeros_like(mask_op, dtype=bool)
    n_op            = 0
    for rid in range(1, n_c + 1):
        region = labeled_op == rid
        if min_area_km2 <= region.sum() * pixel_area_km2 <= max_area_km2:
            final_op[region] = True
            n_op += 1
    mask_ext = final_op & (rain_mm_h >= extreme_thr)
    if mask_ext.sum() == 0:
        return final_op, mask_ext, n_op, 0
    mask_ext = closing(mask_ext, disk(2))
    mask_ext = remove_small_objects(mask_ext, min_size=50)
    _, n_ext = label(mask_ext)
    return final_op, mask_ext, n_op, n_ext


def _render_image(ax, img_mm, storm_op, storm_ext, n_op, n_ext):
    ax.set_facecolor(BG_COLOR)
    im = ax.imshow(np.ma.masked_invalid(img_mm),
                   cmap=RAIN_CMAP, norm=RAIN_NORM, interpolation="nearest")
    if n_op > 0:
        ov = np.zeros((*storm_op.shape, 4))
        ov[storm_op] = [1.0, 0.6, 0.0, 0.25]
        ax.imshow(ov, interpolation="nearest")
    if n_ext > 0:
        ov = np.zeros((*storm_ext.shape, 4))
        ov[storm_ext] = [1.0, 0.0, 1.0, 0.35]
        ax.imshow(ov, interpolation="nearest")
    ax.axis("off")
    return im


def save_gt_grid(samples, cfg, out_path):
    """
    samples: list of dicts, each with keys 'dt', 'gt_mm' (dict h->array),
             'masks' (dict h->array)
    """
    horizons = cfg.horizons
    n_rows   = len(samples)
    n_cols   = len(horizons)
    metrics_flag = getattr(cfg, 'show_storm_metrics', False)

    FIG_BG    = "white"
    TEXT_DARK = "#111111"
    GT_COLOR  = "#333333"

    IMG_W = 4.0
    ROW_H = 4.0
    CB_W  = 0.35

    FIG_W = IMG_W * n_cols + CB_W
    FIG_H = ROW_H * n_rows

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=FIG_BG)

    gs = gridspec.GridSpec(
        n_rows, n_cols + 1,
        figure=fig,
        width_ratios=[1.0] * n_cols + [CB_W / IMG_W],
        hspace=0.16,
        wspace=0.04,
        left=0.02, right=0.98,
        top=0.88,  bottom=0.10,
    )

    fig.suptitle(cfg.title, color=TEXT_DARK, fontsize=22, fontweight="bold", y=0.97)

    last_im = None

    for r_idx, sd in enumerate(samples):
        dt    = sd['dt']
        gt_mm = sd['gt_mm']
        masks = sd['masks']
        target_times = {h: dt + timedelta(minutes=h) for h in horizons}

        for c_idx, h in enumerate(horizons):
            valid   = masks[h] > 0.5
            gt_data = np.where(valid & (gt_mm[h] >= 0.1), gt_mm[h], np.nan)
            op, ext, n_op, n_ext = detect_storms_two_level(
                np.where(valid, gt_mm[h], np.nan),
                cfg.operational_thr, cfg.extreme_thr,
                cfg.storm_min_pixels, cfg.morphology_disk_size,
                cfg.storm_min_area_km2, cfg.storm_max_area_km2, cfg.pixel_area_km2,
            )
            p99 = (float(np.percentile(gt_data[np.isfinite(gt_data)], 99))
                   if np.isfinite(gt_data).any() else 0.0)

            ax = fig.add_subplot(gs[r_idx, c_idx])
            im = _render_image(ax, gt_data, op, ext, n_op, n_ext)
            last_im = im

            ts = target_times[h]
            line1 = f"t+{h} min"
            line2 = (f"P99={p99:.1f}  Conv={n_op}  Ext={n_ext}"
                     if metrics_flag else ts.strftime('%Y-%m-%d %H:%M'))

            if r_idx == 0:
                ax.set_title(f"{line1}\n{line2}", color=GT_COLOR,
                             fontsize=16, fontweight="bold", pad=10)

            for sp in ax.spines.values():
                sp.set_edgecolor(GT_COLOR)
                sp.set_linewidth(2.0)

    if last_im is not None:
        cbar_ax = fig.add_subplot(gs[:, -1])
        cbar_ax.set_facecolor(FIG_BG)
        cbar = fig.colorbar(last_im, cax=cbar_ax, ticks=RAIN_LEVELS, spacing="proportional")
        cbar.outline.set_edgecolor("#999999"); cbar.outline.set_linewidth(0.5)
        cbar_ax.yaxis.set_ticks_position("left")
        cbar_ax.yaxis.set_label_position("left")
        cbar_ax.tick_params(axis="y", length=0, pad=6)
        plt.setp(cbar_ax.get_yticklabels(), color=TEXT_DARK, fontsize=13,
                 fontfamily="monospace", fontweight="bold")
        cbar_ax_r = cbar_ax.twinx()
        cbar_ax_r.set_ylim(cbar_ax.get_ylim()); cbar_ax_r.set_yticks([])
        cbar_ax_r.set_ylabel("mm / h", color=GT_COLOR, fontsize=14, fontweight="bold",
                              labelpad=14, rotation=270, va="bottom")
        cbar_ax_r.spines[:].set_visible(False)
        for thr, badge, col in [
            (cfg.operational_thr, f"OPR\n{cfg.operational_thr:.0f}", "#cc9900"),
            (cfg.extreme_thr,     f"EXT\n{cfg.extreme_thr:.0f}",     "#cc5500"),
        ]:
            cbar_ax.axhline(y=thr, color=col, linewidth=1.8, linestyle="--", alpha=0.9,
                            xmin=-0.5, xmax=1.2, clip_on=False)
            cbar_ax.annotate(badge, xy=(1.15, thr), xycoords=("axes fraction", "data"),
                             fontsize=11, fontweight="bold", color=col, va="center", ha="left",
                             bbox=dict(boxstyle="round,pad=0.3", fc=FIG_BG, ec=col, lw=1.2))

    legend_patches = [
        mpatches.Patch(facecolor=GT_COLOR, label="Ground Truth Radar"),
        mpatches.Patch(facecolor=(1.0, 0.6, 0.0, 0.4), label=f"Operational >= {cfg.operational_thr:.0f} mm/h"),
        mpatches.Patch(facecolor=(1.0, 0.0, 1.0, 0.5), label=f"Extreme >= {cfg.extreme_thr:.0f} mm/h"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=len(legend_patches),
               fontsize=14, framealpha=0.9, facecolor="white", edgecolor="#999999",
               labelcolor=TEXT_DARK, bbox_to_anchor=(0.5, 0.0))

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
    cfg = GTGridConfig()
    os.makedirs(cfg.out_dir, exist_ok=True)

    print(f"Loading metadata: {cfg.metadata_csv}")
    df     = pd.read_csv(cfg.metadata_csv)
    df_top = df.nlargest(cfg.num_samples, 'p99(x_seq)').reset_index(drop=True)

    print(f"Top {cfg.num_samples} samples by P99:")
    print(df_top[['datetime', 'p99(x_seq)']].to_string(index=False))
    print(f"Horizons   : {cfg.horizons}")

    samples = []
    for idx, row in df_top.iterrows():
        print(f"Loading sample [{idx+1}/{cfg.num_samples}]  datetime={row['datetime']}")
        targets_np, masks_np, dt = load_ground_truth_sample(row, cfg)
        gt_mm = {h: targets_np[h_idx] for h_idx, h in enumerate(cfg.horizons)}
        masks = {h: masks_np[h_idx]   for h_idx, h in enumerate(cfg.horizons)}
        samples.append({'dt': dt, 'gt_mm': gt_mm, 'masks': masks})

    out_path = os.path.join(cfg.out_dir, cfg.out_fname)
    save_gt_grid(samples, cfg, out_path)

    print(f"Done. Grid saved to: {out_path}")


if __name__ == "__main__":
    main()