"""
Pre-sample ALL Metadata (Train, Val, Test) - Exact Zarr Approach
==================================================================

This script pre-samples ALL metadata splits (train, val, test) using the same
two-bucket quantile threshold approach, exactly matching the Zarr code behavior.

Run this ONCE before training to create:
- metadata/train_sampled.csv
- metadata/val_sampled.csv  
- metadata/test_sampled.csv

Usage:
    python presample_all_metadata.py

Then run your training script with USE_PRESAMPLED_ALL=True
"""

import os
import argparse
import numpy as np
import pandas as pd


def presample_metadata(
    input_csv,
    output_csv, 
    split_name,
    p99_top_ratio=0.15,
    prob=0.95,
    seed=1
):
    """
    Pre-sample metadata using two-bucket quantile threshold approach.
    
    This exactly replicates the sampling logic from create_zarr_dataset.py
    
    Parameters
    ----------
    input_csv : str
        Path to input metadata CSV (e.g., "metadata/train.csv")
    output_csv : str
        Path to output sampled CSV (e.g., "metadata/train_sampled.csv")
    split_name : str
        Name of split for display (e.g., "train", "val", "test")
    p99_top_ratio : float
        Fraction of samples to treat as "top" severe rain (default: 0.15 = top 15%)
    prob : float
        Sampling probability for top bucket (default: 0.95 = keep 95% of severe rain)
        Rest bucket is sampled with probability (1 - prob)
    seed : int
        Random seed for reproducibility
    """
    print("=" * 80)
    print(f"PRE-SAMPLING {split_name.upper()} METADATA")
    print("=" * 80)
    print(f"Input:  {input_csv}")
    print(f"Output: {output_csv}")
    print(f"Parameters: p99_top_ratio={p99_top_ratio}, prob={prob}, seed={seed}")
    print("=" * 80 + "\n")
    
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    
    print(f"Loading metadata from: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df):,} samples\n")

    if 'p99(x)' not in df.columns:
        raise ValueError(
            f"Column 'p99(x)' not found in {input_csv}\n"
            f"Available columns: {list(df.columns)}"
        )

    s = df["p99(x)"].to_numpy()

    valid = np.isfinite(s)  

    n_bad = (~valid).sum()

    if n_bad > 0:
        print(f"⚠️  Removing {n_bad} samples with invalid p99(x)")

    df = df[valid].reset_index(drop=True)
    s = s[valid]

    print(f"Remaining samples after cleaning: {len(df):,}\n")

    p99_top_ratio = float(np.clip(p99_top_ratio, 0.0, 1.0))
    q = 1.0 - p99_top_ratio
    thr = np.quantile(s, q) if 0.0 < q < 1.0 else (np.min(s) if q <= 0 else np.max(s))
    
    top_mask = (s >= thr)
    rng = np.random.RandomState(seed) if seed is not None else np.random
    rand = rng.rand(len(s))
    
    keep_mask = np.where(top_mask, rand < prob, rand < (1.0 - prob))
    indices = np.nonzero(keep_mask)[0]
    
    n_top = int(top_mask.sum())
    n_rest = len(s) - n_top
    n_top_kept = int((top_mask & keep_mask).sum())
    n_rest_kept = int((~top_mask & keep_mask).sum())
    
    print("SAMPLING STATISTICS:")
    print("-" * 80)
    print(f"Quantile cutoff:     q={q:.3f} → threshold={thr:.6f}")
    print(f"Original samples:    top={n_top:,} | rest={n_rest:,} | total={len(s):,}")
    print(f"Sampling probs:      top={prob:.2f} | rest={1.0-prob:.2f}")
    print(f"Kept samples:        top={n_top_kept:,} | rest={n_rest_kept:,} | total={len(indices):,}")
    print(f"Keep rate:           {len(indices)/len(s)*100:.2f}%")
    print(f"Top ratio (kept):    {n_top_kept/len(indices)*100:.2f}% (vs {p99_top_ratio*100:.1f}% original)")
    print("-" * 80 + "\n")
    
    df_sampled = df.iloc[indices].copy()

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_sampled.to_csv(output_csv, index=False)
    
    print(f"Saved sampled {split_name} metadata to: {output_csv}\n")
    
    return df_sampled


def presample_all_splits(
    p99_top_ratio=0.15,
    prob=0.95,
    train_seed=1,
    val_seed=2,
    test_seed=3
):
    """
    Pre-sample all three splits (train, val, test).
    
    Uses different seeds for each split to ensure different random sampling.
    
    Parameters
    ----------
    p99_top_ratio : float
        Top ratio for all splits (default: 0.15)
    prob : float
        Sampling probability for all splits (default: 0.95)
    train_seed : int
        Seed for train split (default: 1)
    val_seed : int
        Seed for val split (default: 2)
    test_seed : int
        Seed for test split (default: 3)
    """
    print("\n" + "=" * 80)
    print("PRE-SAMPLING ALL SPLITS (TRAIN, VAL, TEST)")
    print("Exact Zarr Approach: Apply sampling to all splits")
    print("=" * 80 + "\n")
    
    splits = [
        ("train", "metadata/train.csv", "metadata/train_sampled.csv", train_seed),
        ("val", "metadata/val.csv", "metadata/val_sampled.csv", val_seed),
        ("test", "metadata/test.csv", "metadata/test_sampled.csv", test_seed),
    ]
    
    results = {}
    
    for split_name, input_csv, output_csv, seed in splits:
        df_sampled = presample_metadata(
            input_csv=input_csv,
            output_csv=output_csv,
            split_name=split_name,
            p99_top_ratio=p99_top_ratio,
            prob=prob,
            seed=seed
        )
        results[split_name] = len(df_sampled)
    
    print("\n" + "=" * 80)
    print("SUMMARY - ALL SPLITS SAMPLED")
    print("=" * 80)
    for split_name, count in results.items():
        print(f"  {split_name:5s}: {count:,} samples (sampled)")
    print("\n✓ All splits pre-sampled successfully!")
    print("✓ Ready to train with USE_PRESAMPLED_ALL=True")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-sample ALL metadata splits (train, val, test) - Exact Zarr approach"
    )
    parser.add_argument(
        "--p99_top_ratio",
        type=float,
        default=0.30,
        help="Top ratio for severe rain (default: 0.15 = top 15%%)"
    )
    parser.add_argument(
        "--prob",
        type=float,
        default=0.90,
        help="Sampling probability for top bucket (default: 0.95)"
    )
    parser.add_argument(
        "--train_seed",
        type=int,
        default=1,
        help="Random seed for train split (default: 1)"
    )
    parser.add_argument(
        "--val_seed",
        type=int,
        default=2,
        help="Random seed for val split (default: 2)"
    )
    parser.add_argument(
        "--test_seed",
        type=int,
        default=3,
        help="Random seed for test split (default: 3)"
    )
    
    args = parser.parse_args()
    
    presample_all_splits(
        p99_top_ratio=args.p99_top_ratio,
        prob=args.prob,
        train_seed=args.train_seed,
        val_seed=args.val_seed,
        test_seed=args.test_seed
    )


if __name__ == "__main__":
    main()