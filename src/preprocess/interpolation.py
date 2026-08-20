import os
import argparse
import numpy as np
from tqdm import tqdm
from affine import Affine
from scipy.interpolate import griddata
from concurrent.futures import ProcessPoolExecutor, as_completed
import wradlib as wrl
from pyproj import CRS, Transformer


nrows, ncols = 1100, 900
radolan_xy = wrl.georef.get_radolan_grid(nrows, ncols)  
target_x = radolan_xy[:, :, 0].ravel()
target_y = radolan_xy[:, :, 1].ravel()
target_shape = (nrows, ncols)

crs_radolan = CRS.from_proj4(
    "+proj=stere +lat_0=90 +lon_0=10 +lat_ts=60 +k=1 +x_0=0 +y_0=0 +ellps=sphere +units=km +no_defs"
)


def interpolate_to_radklim(data, transform_vals, crs_wkt):
    rows, cols = data.shape
    affine = Affine.from_gdal(*transform_vals)

    yy, xx = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    xs, ys = affine * (xx, yy)

    flat_data = data.ravel()
    flat_xs   = xs.ravel()
    flat_ys   = ys.ravel()

    crs_src = CRS.from_wkt(crs_wkt)
    transformer = Transformer.from_crs(crs_src, crs_radolan, always_xy=True)
    xs_reproj, ys_reproj = transformer.transform(flat_xs, flat_ys)

    valid  = np.isfinite(flat_data)
    valid &= np.isfinite(xs_reproj)
    valid &= np.isfinite(ys_reproj)

    if np.sum(valid) < 10:
        return None

    points = np.column_stack((xs_reproj[valid], ys_reproj[valid]))
    values = flat_data[valid].astype(np.float32, copy=False)

    try:
        interpolated = griddata(
            points, values,
            (target_x, target_y),
            method="linear",
            fill_value=np.nan,
        )
    except Exception:
        interpolated = griddata(
            points, values,
            (target_x, target_y),
            method="nearest",
            fill_value=np.nan,
        )

    return interpolated.reshape(target_shape)

def convert_one(npz_path, out_npy_path):
    if os.path.exists(out_npy_path):
        return f"Skipped (exists): {npz_path}"

    try:
        with np.load(npz_path, allow_pickle=True) as f:
            if 'data' not in f or 'transform' not in f or 'crs' not in f:
                return f"Missing required keys in {npz_path}"

            data = f['data'].astype(np.float32)
            data[data == -1000.0] = np.nan

            transform_vals = f['transform']
            crs_wkt = f['crs'].item() if isinstance(f['crs'], np.ndarray) else f['crs']

            result = interpolate_to_radklim(data, transform_vals, crs_wkt)
            if result is None:
                return f"Not enough valid points in {npz_path}"

            os.makedirs(os.path.dirname(out_npy_path), exist_ok=True)
            np.save(out_npy_path, result.astype(np.float16))
            return f"OK: {npz_path} → {result.shape}"

    except Exception as e:
        return f"Error with {npz_path}: {e}"


def satellite_npz_to_npy_all(
    indir,
    outdir,
    limit=None,
    max_workers=8,
):

    print("=" * 70)
    print("SATELLITE REGRIDDING — ALL SEASONS")
    print("=" * 70)
    print(f"Input dir:  {indir}")
    print(f"Output dir: {outdir}")
    print(f"Workers:    {max_workers}")
    print(f"Limit:      {limit if limit is not None else 'None (all files)'}")
    print("=" * 70)

    print("Scanning for .npz files...")
    all_npz = []
    for root, _, files in os.walk(indir):
        for fname in files:
            if fname.endswith(".npz") and ("CH7" in fname or "CH9" in fname):
                all_npz.append(os.path.join(root, fname))

    all_npz = sorted(all_npz)

    if limit is not None:
        all_npz = all_npz[:limit]

    print(f"Total .npz files found: {len(all_npz):,}")

    if len(all_npz) == 0:
        print("No files found! Check indir path and folder structure.")
        return

    est_gb = len(all_npz) * 2 / 1024  
    print(f"Estimated output size: ~{est_gb:.0f} GB")
    print("=" * 70)

    tasks   = []
    skipped = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for npz_path in all_npz:
            rel_path = os.path.relpath(npz_path, indir)
            out_path = os.path.join(outdir, os.path.splitext(rel_path)[0] + ".npy")

            if os.path.exists(out_path):
                skipped += 1
                continue

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            tasks.append(executor.submit(convert_one, npz_path, out_path))

        print(f"Tasks scheduled: {len(tasks):,} | Already skipped: {skipped:,}")
        print("=" * 70)

        errors = 0
        for future in tqdm(as_completed(tasks), total=len(tasks), desc="Regridding"):
            result = future.result()
            if any(result.startswith(k) for k in ["Error", "Missing", "Not enough"]):
                errors += 1
                tqdm.write(f"  WARN: {result}")

    print("=" * 70)
    print(f"Done!")
    print(f"Total processed: {len(tasks):,}")
    print(f"Skipped:         {skipped:,}")
    print(f"Errors:          {errors:,}")
    print(f"Output saved to: {outdir}")
    print("=" * 70)


def verify_output(outdir, n_samples=10):
    """
    Verify output files have correct shape [1100, 900].
    """
    print("\nVerifying output files...")
    npy_files = []
    for root, _, files in os.walk(outdir):
        for f in files:
            if f.endswith(".npy"):
                npy_files.append(os.path.join(root, f))
        if len(npy_files) >= n_samples:
            break

    if len(npy_files) == 0:
        print("No .npy files found in output directory!")
        return

    all_ok = True
    for fpath in npy_files[:n_samples]:
        arr = np.load(fpath)
        ok  = arr.shape == (1100, 900)
        status = "✓" if ok else "✗ WRONG SHAPE"
        print(f"  {status} {os.path.basename(fpath)} → shape {arr.shape}, dtype {arr.dtype}")
        if not ok:
            all_ok = False

    if all_ok:
        print(f"\nAll {min(n_samples, len(npy_files))} sampled files correct ✓")
    else:
        print("\nSome files have wrong shape! Check conversion.")


def get_parser():
    parser = argparse.ArgumentParser(
        description="Regrid ALL satellite .npz files to RADKLIM [1100,900] grid"
    )
    parser.add_argument(
        "--indir",
        type=str,
        default="/home/fe/sajib/scratch/weather-data/satellite_de",
        help="Root directory of raw .npz satellite files",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="/home/fe/sajib/scratch/weather-data/satellite_de_regridded",
        help="Output directory for regridded .npy files",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N files for testing. Default: None (all files)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="After conversion, verify a sample of output file shapes",
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args   = parser.parse_args()

    satellite_npz_to_npy_all(
        indir       = args.indir,
        outdir      = args.outdir,
        limit       = args.limit,
        max_workers = args.workers,
    )

    if args.verify:
        verify_output(args.outdir)