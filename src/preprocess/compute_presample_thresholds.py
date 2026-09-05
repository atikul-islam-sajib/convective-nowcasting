import numpy as np
import pandas as pd


def recommend_sampling_params(
    csv_path: str,
    p99_col: str = 'p99(x_seq)',
    target_thresholds_mmh: list = [0.5, 1.0, 2.0, 5.0],
    prob_values: list = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
    min_ratio: float = 1.5,
    max_ratio: float = 3.0,
    min_n_output: int = 50000,
    min_thr_mmh: float = 1.0,
    ideal_ratio: float = 2.0,
    top_n: int = 3,
    patch_height: int = 256,
    patch_width: int = 256,
    stride_patch: int = 128,
    season_tag: str = 'summer',
):
    df = pd.read_csv(csv_path)
    s  = df[p99_col].dropna().to_numpy()
    n  = len(s)

    print(f"{'='*80}")
    print(f"SAMPLING PARAMETER ANALYSIS")
    print(f"{'='*80}")
    print(f"CSV          : {csv_path}")
    print(f"Total patches: {n:,}")
    print(f"Dry (p99=0)  : {(s==0).sum():,} ({(s==0).mean()*100:.1f}%)")
    print(f"Rainy (p99>0): {(s>0).sum():,} ({(s>0).mean()*100:.1f}%)")
    print(f"{'='*80}\n")

    print(f"{'mm/h':<8} {'p99_top_ratio':<16} {'actual_thr':<14} {'prob':<8} {'n_output':<10} {'heavy%':<10} {'light%':<10} {'ratio'}")
    print("-" * 95)

    results = []

    for thr_mmh in target_thresholds_mmh:
        q             = float(np.mean(s < thr_mmh))
        p99_top_ratio = round(1 - q, 3)
        actual_thr    = float(np.quantile(s, q))

        top_mask = (s >= actual_thr) & (s > 0)
        n_top    = int(top_mask.sum())
        n_rest   = n - n_top

        for prob in prob_values:
            n_top_kept  = int(n_top  * prob)
            n_rest_kept = int(n_rest * (1 - prob))
            n_output    = n_top_kept + n_rest_kept
            if n_output == 0:
                continue
            pct_heavy = n_top_kept  / n_output * 100
            pct_light = n_rest_kept / n_output * 100
            ratio     = n_top_kept  / max(n_rest_kept, 1)

            print(f"{thr_mmh:<8.1f} {p99_top_ratio:<16.3f} {actual_thr:<14.4f} {prob:<8.2f} {n_output:<10,} {pct_heavy:<10.1f} {pct_light:<10.1f} {ratio:.1f}:1")

            results.append({
                'thr_mmh'      : thr_mmh,
                'p99_top_ratio': p99_top_ratio,
                'actual_thr'   : round(actual_thr, 4),
                'prob'         : prob,
                'n_output'     : n_output,
                'heavy%'       : round(pct_heavy, 1),
                'light%'       : round(pct_light, 1),
                'ratio'        : round(ratio, 1),
            })
        print()

    candidates = [
        r for r in results
        if min_ratio  <= r['ratio']    <= max_ratio
        and r['thr_mmh']               >= min_thr_mmh
        and r['n_output']              >= min_n_output
    ]

    # sort by closest ratio to ideal, then largest dataset
    candidates = sorted(
        candidates,
        key=lambda x: (abs(x['ratio'] - ideal_ratio), -x['n_output'])
    )

    print(f"\n{'='*80}")
    print(f"TOP {top_n} RECOMMENDATIONS")
    print(f"(filter: ratio {min_ratio}-{max_ratio}, thr>={min_thr_mmh}mm/h, n_output>={min_n_output:,})")
    print(f"(sort: closest ratio to {ideal_ratio}:1, then largest dataset)")
    print(f"{'='*80}")

    for i, r in enumerate(candidates[:top_n], 1):
        print(f"\n#{i}  mm/h={r['thr_mmh']}  p99_top_ratio={r['p99_top_ratio']}  prob={r['prob']}")
        print(f"    actual_thr={r['actual_thr']} mm/h")
        print(f"    n_output={r['n_output']:,}  heavy={r['heavy%']}%  light={r['light%']}%  ratio={r['ratio']}:1")
        print(f"    Command:")
        print(f"    python presample_patch_metadata.py \\")
        print(f"        --p99_top_ratio {r['p99_top_ratio']} --prob {r['prob']} \\")
        print(f"        --patch_height {patch_height} --patch_width {patch_width} \\")
        print(f"        --stride_patch {stride_patch} --season_tag {season_tag}")

    return candidates[:top_n]

recommend_sampling_params(
    csv_path              = 'metadata_patch/val_patch_256x256_s128_summer.csv',
    target_thresholds_mmh = [0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0], 
    prob_values           = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
    min_ratio             = 1.5,
    max_ratio             = 4.0,   
    min_n_output          = 50000, 
    min_thr_mmh           = 5.0, 
    ideal_ratio           = 2.0,
    top_n                 = 3,
    patch_height          = 256,
    patch_width           = 256,
    stride_patch          = 64,
    season_tag            = 'summer',
)