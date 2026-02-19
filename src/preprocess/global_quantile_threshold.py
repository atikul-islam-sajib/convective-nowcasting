"""
Compute and save global percentile threshold from RADOLAN YW radar data.
Author: Sajib
Modified: Ultra-defensive version with explicit NaN handling
"""

"""
Global Radar Percentile Threshold Computation (Ultra-Defensive Version).

This module computes a global precipitation intensity threshold from
RADOLAN YW radar data using robust, memory-safe, and aggressively validated
sampling and percentile estimation.

Key features:
    - Chunked file processing to limit memory usage.
    - Parallel loading of radar files.
    - Aggressive NaN/Inf filtering at all stages.
    - Upper/lower bound validation.
    - Reproducible random sampling.
    - Manual percentile computation with verification.
    - Detailed diagnostic logging.

The computed threshold is typically used for storm detection,
clipping, or normalization in downstream ML pipelines.

Outputs:
    - JSON metadata file with threshold information.
    - NPY file containing the scalar threshold value.

Typical usage:
    python compute_global_threshold.py \
        --csv_path metadata/train_sampled.csv \
        --percentile 99.5 \
        --n_jobs 8 \
        --sample_size 10000000
"""

import json
import argparse
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

warnings.filterwarnings("ignore", category=RuntimeWarning)



def load_nonzero_pixels(full_path):
    """Load one radar file and return finite nonzero pixels."""
    try:
        arr = np.load(full_path)

        arr = np.asarray(arr, dtype=np.float32)
        mask = np.isfinite(arr) & (arr > 0) & (arr < 1e6)
        pixels = arr[mask]
        
        if len(pixels) > 0:
            pixels = pixels[np.isfinite(pixels)]
        
        return pixels
    except Exception as e:
        print(f"Error loading {full_path}: {e}")
        return np.array([])


def safe_percentile(data, percentile):
    """
    Compute a percentile using aggressive validation and manual interpolation.

    This function performs multiple validation and filtering passes
    before computing the percentile. It avoids numerical instability
    by manually implementing linear interpolation on sorted data.

    The computed value is cross-validated against NumPy's percentile
    implementation for diagnostic purposes.

    Parameters
    ----------
    data : numpy.ndarray
        One-dimensional array of sampled precipitation values.
    percentile : float
        Percentile to compute (0–100).

    Returns
    -------
    float
        Computed percentile threshold.

    Raises
    ------
    ValueError
        If no valid values remain after filtering, or if the result
        is not finite.
    """
    clean_data = data[np.isfinite(data)]
    
    print(f"\n  Before cleaning: {len(data):,} values")
    print(f"  After cleaning: {len(clean_data):,} values")
    
    if len(clean_data) == 0:
        raise ValueError("No valid data after filtering NaN/Inf!")
    
    nan_count = np.isnan(clean_data).sum()
    inf_count = np.isinf(clean_data).sum()
    
    print(f"  NaN count in 'clean' data: {nan_count}")
    print(f"  Inf count in 'clean' data: {inf_count}")
    
    if nan_count > 0 or inf_count > 0:
        print("  WARNING: Found NaN/Inf after filtering! Filtering again...")
        clean_data = clean_data[np.isfinite(clean_data)]
        print(f"  After second filter: {len(clean_data):,} values")
    
    print(f"  Min: {clean_data.min():.4f}")
    print(f"  Max: {clean_data.max():.4f}")
    print(f"  Mean: {clean_data.mean():.4f}")
    print(f"  Std: {clean_data.std():.4f}")
    
    print(f"  Sorting {len(clean_data):,} values...")
    sorted_data = np.sort(clean_data)
    
    n = len(sorted_data)
    idx = (percentile / 100.0) * (n - 1)
    
    idx_lower = int(np.floor(idx))
    idx_upper = int(np.ceil(idx))
    
    if idx_lower == idx_upper:
        threshold = sorted_data[idx_lower]
    else:
        weight = idx - idx_lower
        threshold = sorted_data[idx_lower] * (1 - weight) + sorted_data[idx_upper] * weight
    
    print(f"    Manual P{percentile} calculation:")
    print(f"    Index: {idx:.2f} (between {idx_lower} and {idx_upper})")
    print(f"    Lower value: {sorted_data[idx_lower]:.4f}")
    print(f"    Upper value: {sorted_data[idx_upper]:.4f}")
    print(f"    Result: {threshold:.4f}")
    
    np_result = np.percentile(sorted_data, percentile)
    print(f"  NumPy percentile: {np_result:.4f}")
    
    threshold = float(threshold)
    
    if not np.isfinite(threshold):
        raise ValueError(f"Computed threshold is not finite: {threshold}")
    
    return threshold


def compute_global_threshold_chunked(csv_path, percentile, n_jobs, sample_size=10_000_000):
    """
        Compute a global radar threshold using chunked parallel processing.

        This function reads radar file paths from a CSV file, loads them in
        batches, filters invalid values, performs controlled random sampling,
        and computes a global percentile threshold.

        The procedure is designed to be memory-safe and resilient to corrupted
        or malformed data.

        Parameters
        ----------
        csv_path : str
            Path to CSV file containing radar file paths.
        percentile : float
            Target percentile (e.g., 95.0, 99.5).
        n_jobs : int
            Number of parallel workers for file loading.
        sample_size : int, optional
            Maximum number of pixels used for percentile estimation.
            Default is 10,000,000.

        Returns
        -------
        tuple
            (threshold, total_pixels)

            threshold : float
                Computed percentile value.
            total_pixels : int
                Total number of valid pixels processed.

        Raises
        ------
        ValueError
            If invalid values remain after final filtering.
        """
    df = pd.read_csv(csv_path)
    
    if "split" in df.columns:
        df = df[df["split"] == "train"]
        print(f"Using only 'train' split: {len(df)} files")
    
    paths = df["radar_path"].values
    batch_size = 500  
    num_batches = (len(paths) + batch_size - 1) // batch_size
    
    print(f"Total files: {len(paths)}")
    print(f"Processing in {num_batches} batches of {batch_size} files each")
    
    all_samples = []
    total_pixels = 0
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(paths))
        batch_paths = paths[start_idx:end_idx]
        
        print(f"\n{'='*60}")
        print(f"Batch {batch_idx + 1}/{num_batches}: Processing {len(batch_paths)} files...")
        
        batch_pixels = Parallel(n_jobs=n_jobs)(
            delayed(load_nonzero_pixels)(p) for p in tqdm(batch_paths, desc=f"Batch {batch_idx+1}")
        )
        
        batch_pixels = [p for p in batch_pixels if len(p) > 0]
        if len(batch_pixels) == 0:
            print("  No valid pixels in this batch, skipping...")
            continue
        
        print("  Concatenating batch arrays...")
        batch_concat = np.concatenate(batch_pixels).astype(np.float64)
        
        print("  Filtering NaN/Inf...")
        valid_mask = np.isfinite(batch_concat) & (batch_concat > 0) & (batch_concat < 1e6)
        batch_concat = batch_concat[valid_mask]
        
        batch_pixel_count = len(batch_concat)
        total_pixels += batch_pixel_count
        
        print(f"  Valid batch pixels: {batch_pixel_count:,}")
        print(f"  Batch min: {batch_concat.min():.2f}")
        print(f"  Batch max: {batch_concat.max():.2f}")
        
        nan_check = np.isnan(batch_concat).sum()
        inf_check = np.isinf(batch_concat).sum()
        if nan_check > 0 or inf_check > 0:
            print(f"  WARNING: Found {nan_check} NaN and {inf_check} Inf after filtering!")
            batch_concat = batch_concat[np.isfinite(batch_concat)]
            print(f"  After re-filter: {len(batch_concat):,} pixels")
        
        sample_from_batch = min(sample_size // num_batches + 1000, batch_pixel_count)
        if batch_pixel_count > sample_from_batch:
            print(f"  Sampling {sample_from_batch:,} pixels from batch...")
            rng = np.random.default_rng(seed=42 + batch_idx)
            batch_sample = rng.choice(batch_concat, size=sample_from_batch, replace=False)
        else:
            batch_sample = batch_concat
        
        batch_sample = batch_sample[np.isfinite(batch_sample)]
        print(f"  Sample size after final filter: {len(batch_sample):,}")
        
        all_samples.append(batch_sample)
        
        del batch_pixels, batch_concat, batch_sample
    
    print(f"\n{'='*60}")
    print(f"Total pixels across all batches: {total_pixels:,}")
    print(f"Number of sample arrays: {len(all_samples)}")
    
    print("\nCombining samples from all batches...")
    combined_samples = np.concatenate(all_samples).astype(np.float64)
    
    print(f"Combined array size: {len(combined_samples):,}")
    print(f"Combined array dtype: {combined_samples.dtype}")
    
    print("\nFinal filtering pass...")
    combined_samples = combined_samples[np.isfinite(combined_samples)]
    combined_samples = combined_samples[combined_samples > 0]
    combined_samples = combined_samples[combined_samples < 1e6]
    
    print(f"After final filters: {len(combined_samples):,}")
    print(f"Sample min: {combined_samples.min():.2f}")
    print(f"Sample max: {combined_samples.max():.2f}")
    print(f"Sample mean: {combined_samples.mean():.2f}")
    
    nan_in_final = np.isnan(combined_samples).sum()
    inf_in_final = np.isinf(combined_samples).sum()
    print(f"NaN in final sample: {nan_in_final}")
    print(f"Inf in final sample: {inf_in_final}")
    
    if nan_in_final > 0 or inf_in_final > 0:
        raise ValueError(f"Found NaN ({nan_in_final}) or Inf ({inf_in_final}) in final sample!")
    
    if len(combined_samples) > sample_size:
        print(f"\nSubsampling to {sample_size:,} pixels...")
        rng = np.random.default_rng(seed=42)
        combined_samples = rng.choice(combined_samples, size=sample_size, replace=False)
        print(f"After subsampling: {len(combined_samples):,} pixels")
    
    print(f"\n{'='*60}")
    print(f"Computing P{percentile} threshold...")
    print('='*60)
    
    threshold = safe_percentile(combined_samples, percentile)
    
    return threshold, total_pixels


def main():
    parser = argparse.ArgumentParser(
        description="Compute and save global storm threshold from radar data"
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="metadata/train_sampled.csv",
        help="Path to CSV file containing radar file paths"
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.5,
        help="Percentile to compute (e.g., 95.0, 99.5)"
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=8,
        help="Number of parallel jobs for loading files"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=10_000_000,
        help="Max number of pixels to use for percentile (default: 10M)"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="metadata/storm_threshold.json",
        help="Output path for JSON file"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("STORM THRESHOLD COMPUTATION (ULTRA-DEFENSIVE VERSION)")
    print("="*60)
    print(f"CSV path: {args.csv_path}")
    print(f"Percentile: {args.percentile}")
    print(f"Parallel jobs: {args.n_jobs}")
    print(f"Sample size: {args.sample_size:,}")
    print("="*60 + "\n")
    
    # Compute threshold
    try:
        thr, total_pixels = compute_global_threshold_chunked(
            args.csv_path, args.percentile, args.n_jobs, args.sample_size
        )
    except Exception as e:
        print(f"\n ERROR: {e}")
        import traceback
        traceback.print_exc()
        return

    info = {
        "percentile": args.percentile,
        "global_threshold": float(thr),
        "unit": "mm/5min",
        "source_csv": args.csv_path,
        "total_pixels": int(total_pixels),
        "sample_size_limit": args.sample_size,
        "sampling_method": "chunked_ultra_defensive",
        "note": "Computed using chunked sampling with manual percentile calculation"
    }
    
    with open(args.out, "w") as f:
        json.dump(info, f, indent=4)

    npy_path = args.out.replace(".json", ".npy")
    np.save(npy_path, thr)
    
    print("\n" + "="*60)
    print(" COMPUTATION COMPLETE")
    print("="*60)
    print(f"Total pixels processed: {total_pixels:,}")
    print(f"P{args.percentile} THRESHOLD = {thr:.4f} mm/5min")
    print(f"Saved JSON: {args.out}")
    print(f"Saved NPY:  {npy_path}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()