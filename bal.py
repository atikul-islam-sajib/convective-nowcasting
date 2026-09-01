"""
Full pipeline:
  1. Crop GT / pySTEPS-LK panels out of the original dark-background PNG
  2. Replace the panel background (RGB 26,26,26) with white
  3. Rebuild the figure in the new layout: rows = method, columns = lead time
  4. Add a fresh colorbar + the exact bottom legend (Ground Truth / PySteps
     Radar / PySteps Multimodal / Operational >=15 / Extreme >=35), matching
     the reference "PySteps Baseline | Radar & Multimodal" figure.

Only requirement: the source PNG at SRC_PNG below.
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import BoundaryNorm, ListedColormap

# ----------------------------------------------------------------------
# 0. CONFIG
# ----------------------------------------------------------------------
SRC_PNG = "/home/sajib/Downloads/last/pySTEPS-Exp/result1320011/PYSTEPS_BASELINE_RESULTS/pysteps_sample_0001_20240630_0100.png"
OUT_PNG = "pysteps_LK_relayout_white.png"

lead_times = ["t+15 min", "t+30 min", "t+45 min", "t+60 min"]
timestamps = ["2024-06-30 01:15", "2024-06-30 01:30", "2024-06-30 01:45", "2024-06-30 02:00"]

OPR_THRESH = 15   # "Operational" threshold, mm/h
EXT_THRESH = 35   # "Extreme" threshold, mm/h

# ----------------------------------------------------------------------
# 1. CROP PANELS OUT OF THE SOURCE PNG
# ----------------------------------------------------------------------
# These pixel ranges were measured from SRC_PNG (1638x2118). If you use a
# different source image, re-measure with the same
# background-color-detection approach (see notes at bottom of file).
row_blocks = [(125, 525), (645, 1045), (1164, 1564), (1684, 2084)]  # image-only, title excluded
col_GT = (546, 823)
col_LK = (1059, 1338)
BG_COLOR = np.array([26, 26, 26])  # figure background color seen through transparent axes


def whiten_bg(crop, tol=18):
    """Replace pixels close to BG_COLOR with white."""
    crop = crop.astype(int)
    diff = np.abs(crop - BG_COLOR).sum(axis=2)
    mask = diff < tol
    out = crop.copy()
    out[mask] = [255, 255, 255]
    return out.astype(np.uint8)


src = np.array(Image.open(SRC_PNG).convert("RGB")).astype(np.uint8)

crops = {}
for i, (y0, y1) in enumerate(row_blocks):
    crops[f"GT_{i}"] = whiten_bg(src[y0:y1, col_GT[0]:col_GT[1]])
    crops[f"LK_{i}"] = whiten_bg(src[y0:y1, col_LK[0]:col_LK[1]])

# ----------------------------------------------------------------------
# 2. PRECIP COLORMAP (matches the original pySTEPS scale)
# ----------------------------------------------------------------------
levels = [0.1, 1, 2, 5, 10, 15, 20, 30, 40, 60, 100]
colors = ["#003a00", "#00a000", "#00e000", "#ffff00", "#ffc000",
          "#ff8000", "#ff4000", "#c00000", "#800080", "#ff00ff"]
cmap = ListedColormap(colors)
norm = BoundaryNorm(levels, cmap.N)

# ----------------------------------------------------------------------
# 3. BUILD THE FIGURE (rows = method, columns = lead time)
# ----------------------------------------------------------------------
rows = [
    ("Ground\nTruth", "GT", "black"),
    ("pySTEPS\nLK", "LK", "#0a6e6e"),
]

n_rows, n_cols = 2, 4
fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.1 * n_cols, 3.3 * n_rows), facecolor="white")

for r, (row_label, key, color) in enumerate(rows):
    for c in range(n_cols):
        ax = axes[r, c]
        ax.imshow(crops[f"{key}_{c}"])
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        prefix = "GT" if key == "GT" else "pySTEPS LK"
        ax.set_title(f"{prefix} {lead_times[c]}\n{timestamps[c]}",
                     fontsize=9, fontweight="bold", color=color)
    axes[r, 0].text(-0.30, 0.5, row_label, transform=axes[r, 0].transAxes,
                     fontsize=12, fontweight="bold", color=color,
                     ha="right", va="center")

fig.suptitle("pySTEPS Lucas-Kanade Optical-Flow Baseline  |  2024-06-30",
              fontsize=14, fontweight="bold")

fig.subplots_adjust(left=0.09, right=0.87, top=0.88, bottom=0.14, wspace=0.05, hspace=0.35)

# ----------------------------------------------------------------------
# 4. COLORBAR WITH OPR / EXT THRESHOLD MARKERS
# ----------------------------------------------------------------------
cbar_ax = fig.add_axes([0.90, 0.16, 0.02, 0.68])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = fig.colorbar(sm, cax=cbar_ax, boundaries=levels, ticks=levels)
cbar.set_label("Precipitation [mm/h]", fontsize=10)
cbar.ax.tick_params(labelsize=8)

for thresh, tag, color in [(OPR_THRESH, "OPR\n15", "sandybrown"),
                            (EXT_THRESH, "EXT\n35", "deeppink")]:
    frac = (thresh - levels[0]) / (levels[-1] - levels[0])
    cbar_ax.axhline(frac, color=color, linewidth=1.5, linestyle="--")
    cbar_ax.annotate(
        tag, xy=(1.6, frac), xycoords=("axes fraction", "data"),
        fontsize=7, color=color, fontweight="bold",
        va="center", ha="left",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=1),
    )

# ----------------------------------------------------------------------
# 5. THE EXACT LEGEND (bottom strip)
# ----------------------------------------------------------------------
legend_handles = [
    mpatches.Patch(facecolor="black", label="Ground Truth"),
    mpatches.Patch(facecolor="teal", label="PySteps Radar"),
    mpatches.Patch(facecolor="#a0522d", label="PySteps Multimodal (Radar+CH7+CH9)"),
    mpatches.Patch(facecolor="navajowhite", label=f"Operational >={OPR_THRESH} mm/h"),
    mpatches.Patch(facecolor="magenta", label=f"Extreme >={EXT_THRESH} mm/h"),
]
fig.legend(
    handles=legend_handles, loc="lower center", ncol=5,
    frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.01),
)

# ----------------------------------------------------------------------
# 6. SAVE
# ----------------------------------------------------------------------
fig.savefig(OUT_PNG, dpi=200, facecolor="white", bbox_inches="tight")
print(f"Saved: {OUT_PNG}")

# ----------------------------------------------------------------------
# NOTES: how row_blocks / col_GT / col_LK were measured
# ----------------------------------------------------------------------
# from PIL import Image
# import numpy as np
# arr = np.array(Image.open(SRC_PNG).convert("RGB")).astype(int)
# bg = np.array([26,26,26])
# colored = (np.abs(arr - bg).sum(axis=2) > 40)
# # column ranges: sum `colored` over a row-slice's y-range, axis=0, find
# #   contiguous runs of True along x
# # row ranges: sum `colored` over a column-slice's x-range, axis=1, find
# #   contiguous runs of True along y