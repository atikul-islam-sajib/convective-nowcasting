import os
import argparse
import numpy as np
import pandas as pd

def presample_metadata(
    input_csv,
    output_csv,
    split_name,
    p99_top_ratio=0.40,
    prob=0.60,
    seed=1,
):
    print("=" * 70)
    print(f"PRE-SAMPLING {split_name.upper()}")
    print("=" * 70)
    print(f"Input:          {input_csv}")
    print(f"Output:         {output_csv}")
    print(f"p99_top_ratio:  {p99_top_ratio} (top {p99_top_ratio*100:.0f}%)")
    print(f"prob:           {prob} (top kept={prob:.2f}, rest kept={1-prob:.2f})")
    print(f"seed:           {seed}")
    print("=" * 70)

    if not os.path.exists(input_csv):
        raise FileNotFoundError(
            f"Input CSV not found: {input_csv}\n"
            f"Run generate_metadata_patch.py first."
        )

    df = pd.read_csv(input_csv)
    print(f"Loaded: {len(df):,} samples")

    col = 'p99(x_seq)'
    if col not in df.columns:
        raise ValueError(
            f"Column '{col}' not found in {input_csv}\n"
            f"Available columns: {list(df.columns)}"
        )

    s = df[col].to_numpy()

    valid = np.isfinite(s)
    n_bad = (~valid).sum()
    if n_bad > 0:
        print(f"Removing {n_bad:,} rows with non-finite p99(x_seq)")
    df = df[valid].reset_index(drop=True)
    s  = s[valid]
    print(f"Clean samples: {len(df):,}")

    p99_top_ratio = float(np.clip(p99_top_ratio, 0.0, 1.0))
    q = 1.0 - p99_top_ratio

    if q <= 0.0:
        thr = float(np.min(s))
    elif q >= 1.0:
        thr = float(np.max(s))
    else:
        thr = float(np.quantile(s, q))

    top_mask = (s >= thr) & (s > 0)
    rng      = np.random.RandomState(seed)
    rand     = rng.rand(len(s))

    keep_mask = np.where(top_mask, rand < prob, rand < (1.0 - prob))
    indices   = np.nonzero(keep_mask)[0]

    n_top          = int(top_mask.sum())
    n_rest         = len(s) - n_top
    n_top_kept     = int((top_mask  & keep_mask).sum())
    n_rest_kept    = int((~top_mask & keep_mask).sum())
    n_total_kept   = len(indices)

    print()
    print("SAMPLING STATISTICS:")
    print("-" * 70)
    print(f"  Threshold:       q={q:.3f} → p99(x_seq) >= {thr:.4f}")
    print(f"  Top bucket:      {n_top:,} samples  → kept {n_top_kept:,}  ({prob*100:.0f}%)")
    print(f"  Rest bucket:     {n_rest:,} samples → kept {n_rest_kept:,}  ({(1-prob)*100:.0f}%)")
    print(f"  Total kept:      {n_total_kept:,} / {len(s):,}  ({n_total_kept/len(s)*100:.2f}%)")
    print(f"  Top ratio kept:  {n_top_kept/n_total_kept*100:.1f}%  (original: {p99_top_ratio*100:.0f}%)")
    print("-" * 70)

    df_sampled = df.iloc[indices].copy()
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)
    df_sampled.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}  ({len(df_sampled):,} rows)")
    print()

    return df_sampled


def presample_all_splits(
    out_dir,
    patch_height,
    patch_width,
    stride_patch,
    season_tag,
    p99_top_ratio,
    prob,
    splits,
    train_seed=1,
    val_seed=2,
    test_seed=3,
):
    print("\n" + "=" * 70)
    print("PATCH METADATA PRE-SAMPLING — ALL SPLITS")
    print("=" * 70)
    print(f"out_dir:        {out_dir}")
    print(f"patch:          {patch_height}x{patch_width}  stride={stride_patch}")
    print(f"season_tag:     {season_tag}")
    print(f"p99_top_ratio:  {p99_top_ratio}")
    print(f"prob:           {prob}")
    print(f"splits:         {splits}")
    print("=" * 70 + "\n")

    seed_map = {
        'train': train_seed,
        'val':   val_seed,
        'test':  test_seed,
    }

    results = {}

    for split in splits:
        base_name = (
            f"{split}_patch_{patch_height}x{patch_width}"
            f"_s{stride_patch}_{season_tag}"
        )
        input_csv  = os.path.join(out_dir, base_name + ".csv")
        output_csv = os.path.join(out_dir, base_name + "_sampled.csv")
        seed       = seed_map.get(split, 1)

        df_sampled = presample_metadata(
            input_csv      = input_csv,
            output_csv     = output_csv,
            split_name     = split,
            p99_top_ratio  = p99_top_ratio,
            prob           = prob,
            seed           = seed,
        )
        results[split] = len(df_sampled)

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for split, count in results.items():
        base_name  = (
            f"{split}_patch_{patch_height}x{patch_width}"
            f"_s{stride_patch}_{season_tag}_sampled.csv"
        )
        print(f"  {split:5s}: {count:>10,} samples → {os.path.join(out_dir, base_name)}")
    print("=" * 70)
    print("Done! Ready for training.")
    print("=" * 70 + "\n")



def get_parser():
    parser = argparse.ArgumentParser(
        description="Pre-sample patch metadata CSVs for nowcasting training"
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default='metadata_patch',
        help="Directory with input CSVs (same as --out_dir in generate_metadata_patch.py)",
    )
    parser.add_argument(
        '--patch_height',
        type=int,
        default=256,
        help="Patch height used in generate_metadata_patch.py (default: 128)",
    )
    parser.add_argument(
        '--patch_width',
        type=int,
        default=256,
        help="Patch width used in generate_metadata_patch.py (default: 128)",
    )
    parser.add_argument(
        '--stride_patch',
        type=int,
        default=64,
        help="Patch stride used in generate_metadata_patch.py (default: 64)",
    )
    parser.add_argument(
        '--season_tag',
        type=str,
        default='all',
        choices=['all', 'summer'],
        help="Season tag used in generate_metadata_patch.py: 'all' or 'summer' (default: all)",
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'val'],
        help="Splits to process (default: train val test)",
    )
    parser.add_argument(
        '--p99_top_ratio',
        type=float,
        default=0.40,
        help="Top bucket ratio: fraction of samples treated as heavy rain (default: 0.40)",
    )
    parser.add_argument(
        '--prob',
        type=float,
        default=0.60,
        help="Keep probability for top bucket (default: 0.60). Rest kept with (1-prob).",
    )
    
    parser.add_argument('--train_seed', type=int, default=1)
    parser.add_argument('--val_seed',   type=int, default=2)
    parser.add_argument('--test_seed',  type=int, default=3)

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args   = parser.parse_args()

    presample_all_splits(
        out_dir       = args.out_dir,
        patch_height  = args.patch_height,
        patch_width   = args.patch_width,
        stride_patch  = args.stride_patch,
        season_tag    = args.season_tag,
        p99_top_ratio = args.p99_top_ratio,
        prob          = args.prob,
        splits        = args.splits,
        train_seed    = args.train_seed,
        val_seed      = args.val_seed,
        test_seed     = args.test_seed,
    )