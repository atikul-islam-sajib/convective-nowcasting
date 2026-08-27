import os
import glob
import warnings

import numpy as np
import matplotlib.pyplot as plt

from satpy import Scene
from pyresample import create_area_def

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from shapely.validation import make_valid


# ============================================================
# SETTINGS
# ============================================================

EUMETSAT_DIR = "/home/sajib/Desktop/convective-nowcasting/eumetsat_downloads"

OUTPUT_DIR = "/home/sajib/Desktop/convective-nowcasting/visualizations"

BBOX = [15, -20, 58, 25]

RESOLUTION = 0.05


# ============================================================
# COUNTRY LABELS
# ============================================================

COUNTRY_LABELS = [
    (30.0, 15.5, "Sudan"),
    (39.0, 15.5, "Eritrea"),
    (42.5, 11.8, "Djibouti"),
    (30.5, 7.5, "South Sudan"),
    (40.0, 9.0, "Ethiopia"),
    (45.5, 5.5, "Somalia"),
    (32.5, 1.5, "Uganda"),
    (37.5, 0.5, "Kenya"),
    (29.9, -1.9, "Rwanda"),
    (29.9, -3.4, "Burundi"),
    (34.8, -6.5, "Tanzania"),
    (25.0, -1.0, "Democratic Republic\nof the Congo"),
]


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


def add_country_borders(ax, color="cyan", linewidth=0.9, zorder=10):
    """
    Draws country borders + coastlines manually instead of using
    ax.add_feature(cfeature.BORDERS)/COASTLINE. Cartopy's add_feature
    transforms every polygon worldwide in one batch, so a single
    malformed geometry anywhere on Earth (Shapely 2.x is strict about
    ring closure; some Natural Earth polygons have tiny floating-point
    mismatches) crashes the whole draw with "Points of LinearRing do
    not form a closed linestring". Loading + repairing geometries one
    at a time and skipping any that are still broken avoids this.
    """
    for kind, category, name in [
        ("countries", "cultural", "admin_0_boundary_lines_land"),
        ("coastline", "physical", "coastline"),
    ]:
        shp_path = shpreader.natural_earth(
            resolution="50m", category=category, name=name
        )
        reader = shpreader.Reader(shp_path)
        n_ok, n_skipped = 0, 0
        for record in reader.records():
            try:
                geom = record.geometry
                if not geom.is_valid:
                    geom = make_valid(geom)
                projected = ax.projection.project_geometry(geom, ccrs.PlateCarree())
                if projected.is_empty:
                    continue
                ax.add_geometries(
                    [projected], crs=ax.projection,
                    facecolor="none", edgecolor=color,
                    linewidth=linewidth, zorder=zorder,
                )
                n_ok += 1
            except Exception:
                n_skipped += 1
        print(f"  ({kind}: {n_ok} drawn, {n_skipped} skipped)")


# ============================================================
# FIND FILES
# ============================================================

nat_files = sorted(
    glob.glob(
        os.path.join(
            EUMETSAT_DIR,
            "**",
            "*.nat"
        ),
        recursive=True
    )
)

print("=" * 70)
print("METEOSAT IR_108 TEST")
print("=" * 70)
print()

print(f"Found {len(nat_files)} files")

if not nat_files:
    raise SystemExit("No .nat files found.")


# ============================================================
# CREATE OUTPUT GRID
# ============================================================

min_lon, min_lat, max_lon, max_lat = BBOX

area_def = create_area_def(
    "east_africa",
    projection="EPSG:4326",
    area_extent=(
        min_lon,
        min_lat,
        max_lon,
        max_lat
    ),
    resolution=(
        RESOLUTION,
        RESOLUTION
    ),
    units="degrees"
)

print()
print("Output grid:")
print(area_def.shape)
print()


# ============================================================
# PROCESS FILES
# ============================================================

for nat_path in nat_files:

    scene_id = os.path.basename(nat_path).replace(
        ".nat",
        ""
    )

    print("=" * 70)
    print(scene_id)
    print("=" * 70)

    try:

        # ----------------------------------------------------
        # LOAD
        # ----------------------------------------------------

        print("Loading...")

        scene = Scene(
            filenames=[nat_path],
            reader="seviri_l1b_native"
        )

        scene.load(["IR_108"])

        print("IR_108 loaded.")

        # ----------------------------------------------------
        # RESAMPLE
        # ----------------------------------------------------

        print("Resampling...")

        with warnings.catch_warnings():

            warnings.simplefilter("ignore")

            resampled_scene = scene.resample(
                area_def,
                resampler="nearest"
            )

        print("Resampling finished.")

        # ----------------------------------------------------
        # GET DATA
        # ----------------------------------------------------

        data = resampled_scene["IR_108"].values

        if np.ma.isMaskedArray(data):
            data = data.filled(np.nan)

        # VERY IMPORTANT:
        # Force the data to a real NumPy array.
        data = np.asarray(
            data,
            dtype=np.float64
        )

        print()
        print("DATA INFORMATION")
        print("-----------------")
        print("Shape:", data.shape)
        print("dtype:", data.dtype)

        print(
            "NaN:",
            np.isnan(data).sum()
        )

        print(
            "Finite:",
            np.isfinite(data).sum()
        )

        print(
            "Minimum:",
            np.nanmin(data)
        )

        print(
            "Maximum:",
            np.nanmax(data)
        )

        print(
            "Mean:",
            np.nanmean(data)
        )

        # ----------------------------------------------------
        # FORCE BAD VALUES TO NaN
        # ----------------------------------------------------

        data[
            (data < 150) |
            (data > 350)
        ] = np.nan

        print()
        print("AFTER FILTERING")
        print("----------------")

        print(
            "Minimum:",
            np.nanmin(data)
        )

        print(
            "Maximum:",
            np.nanmax(data)
        )

        # Use percentile limits (shared by all tests below)
        vmin = np.nanpercentile(
            data,
            1
        )

        vmax = np.nanpercentile(
            data,
            99
        )

        # ====================================================
        # GEOGRAPHIC MAP (with country borders + labels)
        # ====================================================

        print()
        print("Creating geographic map...")

        fig = plt.figure(
            figsize=(12, 9)
        )

        ax = plt.axes(
            projection=ccrs.PlateCarree()
        )

        # ----------------------------------------------------
        # PLOT DATA
        # ----------------------------------------------------

        image = ax.imshow(
            data,
            origin="lower",
            extent=[
                min_lon,
                max_lon,
                min_lat,
                max_lat
            ],
            transform=ccrs.PlateCarree(),
            cmap="inferno_r",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            aspect="auto",
            zorder=1
        )

        # ----------------------------------------------------
        # BORDERS
        # ----------------------------------------------------

        # ----------------------------------------------------
        # BORDERS + COASTLINE (manual, crash-resistant version)
        # ----------------------------------------------------

        add_country_borders(ax, color="cyan", linewidth=0.9, zorder=10)

        # ----------------------------------------------------
        # COUNTRY LABELS
        # ----------------------------------------------------

        for lon, lat, name in COUNTRY_LABELS:

            ax.text(
                lon,
                lat,
                name,
                transform=ccrs.PlateCarree(),
                fontsize=9,
                fontweight="bold",
                ha="center",
                va="center",
                color="black",
                bbox=dict(
                    facecolor="white",
                    alpha=0.75,
                    edgecolor="black",
                    linewidth=0.4,
                    pad=1.5
                ),
                zorder=20
            )

        # ----------------------------------------------------
        # GRID — disabled. Cartopy 0.25.0's gridliner has a known
        # internal bug (unrelated to our data or borders code) that
        # throws "LinearRing not closed" while positioning the title
        # against gridline labels. Not essential to the requirement
        # (country names + borders), so skipping it entirely.
        # ----------------------------------------------------

        # ----------------------------------------------------
        # EXTENT — set LAST, after all plotting/features, so
        # imshow/add_feature autoscaling can't override it.
        # ----------------------------------------------------

        ax.set_extent(
            [min_lon, max_lon, min_lat, max_lat],
            crs=ccrs.PlateCarree()
        )

        # ----------------------------------------------------
        # COLORBAR
        # ----------------------------------------------------

        cbar = fig.colorbar(
            image,
            ax=ax,
            shrink=0.8,
            pad=0.04
        )

        cbar.set_label(
            "Brightness temperature (K)"
        )

        # ----------------------------------------------------
        # TITLE
        # ----------------------------------------------------

        ax.set_title(
            f"Meteosat SEVIRI IR 10.8 um\n{scene_id}",
            fontsize=12
        )

        # ----------------------------------------------------
        # SAVE
        # NOTE: bbox_inches="tight" removed — it triggers a
        # known Cartopy/Matplotlib bug that can crop the whole
        # GeoAxes (including borders/labels) away, leaving only
        # the colorbar. fig.tight_layout() is used instead.
        # ----------------------------------------------------

        fig.tight_layout()

        map_file = os.path.join(
            OUTPUT_DIR,
            f"{scene_id}_map.png"
        )

        fig.savefig(
            map_file,
            dpi=150,
            facecolor="white"
        )

        plt.close(fig)

        print()
        print("GEOGRAPHIC MAP SAVED:")
        print(map_file)

        print()
        print("DONE")

    except Exception as e:

        print()
        print("ERROR:")
        print(
            type(e).__name__,
            str(e)
        )
        import traceback
        traceback.print_exc()

        plt.close("all")


# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 70)
print("ALL FILES FINISHED")
print("=" * 70)
print()
print(
    "Output directory:"
)
print(
    OUTPUT_DIR
)