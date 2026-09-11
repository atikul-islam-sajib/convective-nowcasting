import argparse
import itertools
import json
import os
import subprocess
import sys
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)

MAIN_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pysteps_lk_extrapolation_fixed.py')

GRID = {
    'motion_method': ['LK'],
    'nowcast_method': ['extrapolation', 'sprog'],
    'extrap_interp_order': [0, 1],
    'disable_persistence_blend': [False, True],
}

QUICK_GRID = {
    'motion_method': ['LK'],
    'nowcast_method': ['extrapolation'],
    'extrap_interp_order': [0, 1],
    'disable_persistence_blend': [False],
}

SCORE_HORIZONS = [15, 30]
SCORE_METRICS = ['CSI@10', 'ETS@10']


def build_configs(grid):
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    configs = []
    for combo in combos:
        cfg = dict(zip(keys, combo))
        configs.append(cfg)
    return configs


def config_to_args(cfg, out_dir, manifest_path, trial_num_processes=1):
    args = [
        sys.executable, MAIN_SCRIPT,
        '--out_dir', out_dir,
        '--subset_manifest_csv', manifest_path,
        '--num_processes', str(trial_num_processes),
        '--motion_method', cfg['motion_method'],
        '--nowcast_method', cfg['nowcast_method'],
        '--extrap_interp_order', str(cfg['extrap_interp_order']),
        '--lk_num_frames', '4',
    ]
    if cfg['disable_persistence_blend']:
        args.append('--disable_persistence_blend')
    return args


def config_label(cfg):
    parts = [f"{k}={v}" for k, v in cfg.items()]
    return ' '.join(parts)


def score_run(mean_metrics_csv):
    if not os.path.exists(mean_metrics_csv):
        return None, None
    df = pd.read_csv(mean_metrics_csv)
    sub = df[(df['model'] == SCORE_MODEL) & (df['horizon_min'].isin(SCORE_HORIZONS))]
    if sub.empty:
        return None, None
    missing = [m for m in SCORE_METRICS if m not in sub.columns]
    if missing:
        print(f'  [WARN] mean_metrics_csv is missing expected columns {missing} -- '
              f'this validation run may predate a script update. Re-run with the '
              f'current pysteps_lk_extrapolation_fixed.py.')
        return None, sub
    score = sub[SCORE_METRICS].mean().mean()
    return float(score), sub


def main():
    parser = argparse.ArgumentParser(description='Grid-search tuning driver for the pySTEPS LK/S-PROG baseline')
    parser.add_argument('--val_size', type=int, default=30, help='Number of sequences in the shared validation subset (default: 30). Keep this smaller than your final test subset.')
    parser.add_argument('--fresh_val_subset', action='store_true', help='Sample a new validation subset from scratch (only needed the first time, or if you want a different val_size).')
    parser.add_argument('--val_manifest', type=str, default='MEAN_PYSTEPS/tuning_val_manifest.csv', help='Path to the shared validation manifest (default: MEAN_PYSTEPS/tuning_val_manifest.csv). All grid configs are scored on the exact same sequences in this file.')
    parser.add_argument('--tune_out_root', type=str, default='LK_EXTRAPOLATION_TUNING', help='Root output folder; each config gets its own subfolder under here.')
    parser.add_argument('--trial_num_processes', type=int, default=8, help='Worker processes used WITHIN each trial (default: 8). Each trial processes val_size samples in parallel across this many processes -- raise this toward your core count for much faster sweeps (trials still run one at a time, but each one is now parallel internally instead of single-threaded).')
    parser.add_argument('--quick', action='store_true', help='Run a tiny 2-config smoke test of the harness itself instead of the full grid -- use this first to confirm everything wires up before committing to the full sweep.')
    args = parser.parse_args()

    grid = QUICK_GRID if args.quick else GRID
    configs = build_configs(grid)
    print(f'Tuning grid: {len(configs)} configurations to evaluate on {args.val_size} validation samples each.\n')

    # --- Step 1: build the shared validation manifest once, if needed ---
    if args.fresh_val_subset or not os.path.exists(args.val_manifest):
        print(f'Sampling a fresh {args.val_size}-sequence validation subset -> {args.val_manifest}')
        bootstrap_cmd = [
            sys.executable, MAIN_SCRIPT,
            '--subset_size', str(args.val_size),
            '--fresh_subset',
            '--subset_manifest_csv', args.val_manifest,
            '--out_dir', os.path.join(args.tune_out_root, '_bootstrap'),
            '--num_processes', '1',
            '--quick_test', '1', 
        ]
        subprocess.run(bootstrap_cmd, check=True)
    else:
        print(f'Reusing existing validation manifest: {args.val_manifest}')
    results = []
    for i, cfg in enumerate(configs):
        label = config_label(cfg)
        out_dir = os.path.join(args.tune_out_root, f'trial_{i:03d}')
        print(f'\n[{i + 1}/{len(configs)}] {label}')
        print(f'    -> {out_dir}')
        cmd = config_to_args(cfg, out_dir, args.val_manifest, trial_num_processes=args.trial_num_processes)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f'    return code {result.returncode} (see output above)')
            results.append({'config': cfg, 'label': label, 'score': None, 'out_dir': out_dir})
            continue
        mean_csv = os.path.join(out_dir, 'lk_extrapolation_radar_multimodal_mean_metrics.csv')
        score, sub = score_run(mean_csv)
        if score is None:
            print('    could not compute a score for this run (see mean_metrics_csv)')
        else:
            print(f'    score (mean of {SCORE_METRICS} @ t+{SCORE_HORIZONS}, {SCORE_MODEL}) = {score:.4f}')
        results.append({'config': cfg, 'label': label, 'score': score, 'out_dir': out_dir})

    results_sorted = sorted(
        [r for r in results if r['score'] is not None],
        key=lambda r: r['score'], reverse=True,
    )
    failed = [r for r in results if r['score'] is None]

    print('\n' + '=' * 80)
    print('TUNING RESULTS  (ranked best -> worst)')
    print('=' * 80)
    for rank, r in enumerate(results_sorted, start=1):
        marker = ' <-- BEST' if rank == 1 else ''
        print(f'{rank:>2}. score={r["score"]:.4f}  {r["label"]}{marker}')
    if failed:
        print(f'\n{len(failed)} config(s) failed or produced no score:')
        for r in failed:
            print(f'   - {r["label"]}  (see {r["out_dir"]})')

    if results_sorted:
        best = results_sorted[0]
        print('\n' + '-' * 80)
        print('BEST CONFIG')
        print('-' * 80)
        print(json.dumps(best['config'], indent=2))
        print(f'\nTo run this config on your real test set:')
        cmd_str = ' '.join(config_to_args(best['config'], 'LK_EXTRAPOLATION_PYSTEPS_FINAL', '<your_final_test_manifest.csv>'))
        cmd_str = cmd_str.replace(f'--num_processes 1', '--num_processes 8')
        print(f'  {cmd_str}')
        print('\n(remember: point --subset_manifest_csv at a DIFFERENT, larger manifest')
        print(' than the tuning validation set above -- don\'t report final numbers on')
        print(' the same samples you tuned on.)')

    summary_path = os.path.join(args.tune_out_root, 'tuning_summary.csv')
    os.makedirs(args.tune_out_root, exist_ok=True)
    pd.DataFrame([{**r['config'], 'score': r['score'], 'out_dir': r['out_dir']} for r in results]).to_csv(summary_path, index=False)
    print(f'\nFull results written to: {summary_path}')


if __name__ == '__main__':
    main()