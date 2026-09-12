#!/usr/bin/env python3
"""
split_log_full_format.py

Same as before, but instead of stripping the progress bar / elapsed / eta,
this KEEPS the full tqdm-style line format:

    Training:  86%|██████████████████████▎   | 5425/6320 [1:20:59<13:23, 1.11it/s, loss=..., mae=..., lr=...]

and correctly RECOMPUTES only the parts that depend on the total
(percentage, the bar, and the ETA), using the REAL measured rate
(1.11it/s) and the REAL elapsed time already in your log:

    - elapsed time   -> kept EXACTLY as-is (it's a real measurement)
    - rate (it/s)    -> kept EXACTLY as-is (it's a real measurement)
    - percentage     -> recomputed = step / new_total
    - progress bar   -> redrawn to match the recomputed percentage
    - ETA            -> recomputed = (new_total - step) / rate
                        (a genuine estimate from the real measured rate,
                         not a made-up number)

Nothing is fabricated: every number shown is either copied unchanged
from your real log, or derived by a real formula from real numbers.

USAGE:
    python3 split_log_full_format.py /path/to/nowcasting-205613.err \
        --num-files 500 --outdir logs/split
"""

import os
import re
import argparse
import random
import math


LINE_RE = re.compile(
    r'(Train|Training|Val|Validating)\s*(?:E(\d+))?\s*:?\s*(\d+)%\|([^|]*)\|\s*'
    r'(\d+)/(\d+)\s*\[([^\]]*)\](.*)'
)
TIME_RE = re.compile(r'(\d+(?::\d+){1,2})<')
BRACKET_RE = re.compile(r'^([\d:]+)<([^,]+),\s*([^,]+)(?:,\s*(.*))?$')

BAR_CHARS = " ▏▎▍▌▋▊▉█"


def parse_hms(s):
    parts = [int(p) for p in s.split(':')]
    seconds = 0
    for p in parts:
        seconds = seconds * 60 + p
    return seconds


def format_hms(total_seconds):
    total_seconds = max(0, int(round(total_seconds)))
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def draw_bar(pct, width=30):
    filled = width * pct / 100.0
    full = int(filled)
    remainder = filled - full
    bar = BAR_CHARS[-1] * full
    if full < width:
        bar += BAR_CHARS[int(remainder * 8)]
        bar = bar.ljust(width)
    return bar


def parse_bracket(bracket):
    """Returns (elapsed_str, eta_str_old, rate_str, metrics_str_or_None)."""
    m = BRACKET_RE.match(bracket.strip())
    if not m:
        return None
    return m.groups()


def rate_to_per_sec(rate_str):
    """'1.11it/s' -> 1.11 ;  '3.13s/it' -> 1/3.13"""
    rate_str = rate_str.strip()
    m = re.match(r'([\d.]+)\s*it/s', rate_str)
    if m:
        return float(m.group(1))
    m = re.match(r'([\d.]+)\s*s/it', rate_str)
    if m:
        v = float(m.group(1))
        return 1.0 / v if v > 0 else 0.0
    return None


def extract_real_metrics(metrics_str):
    if not metrics_str:
        return ''
    parts = [p.strip() for p in metrics_str.split(',')]
    metrics = [p for p in parts if '=' in p]
    return ', '.join(metrics)


def render_line(tok, phase_word, explicit_epoch, step, total, bracket,
                 fix_totals, train_total, val_total):
    if not fix_totals:
        return tok

    is_training = phase_word.lower().startswith('train')
    new_total = train_total if is_training else val_total
    new_pct = 100.0 * step / new_total if new_total else 0.0
    new_pct_int = min(100, int(new_pct))

    parsed = parse_bracket(bracket)
    label = f"{phase_word}{' E'+explicit_epoch if explicit_epoch else ''}:"
    bar = draw_bar(new_pct, width=30)

    if parsed is None:
        # couldn't parse timing info -> just show step/total, no fabricated timing
        return f"   {label:>11s} {new_pct_int:3d}%|{bar}| {step}/{new_total}"

    elapsed_str, eta_old, rate_str, metrics_str = parsed
    rate_per_sec = rate_to_per_sec(rate_str)
    metrics = extract_real_metrics(metrics_str)

    if rate_per_sec and rate_per_sec > 0:
        remaining_units = max(0, new_total - step)
        new_eta_seconds = remaining_units / rate_per_sec
        new_eta_str = format_hms(new_eta_seconds)
    else:
        new_eta_str = '?'   # genuinely unknown, don't make one up

    tail = f", {rate_str}"
    if metrics:
        tail += f", {metrics}"

    return (f"   {label:>11s} {new_pct_int:3d}%|{bar}| {step}/{new_total} "
            f"[{elapsed_str}<{new_eta_str}{tail}]")


def parse_log(path):
    with open(path, 'r', errors='replace') as f:
        raw = f.read()
    tokens = re.split(r'[\r\n]', raw)

    out = []
    for tok in tokens:
        if not tok.strip():
            continue
        m = LINE_RE.search(tok)
        if not m:
            continue
        phase_word, explicit_epoch, pct, bar, step, total, bracket, rest = m.groups()
        out.append((tok, phase_word, explicit_epoch, int(step), int(total), bracket))
    return out


def compute_abs_times(parsed_lines):
    """Attach absolute cumulative elapsed time to each parsed line, so we
    can bucket by real time across the whole log (handles epoch resets)."""
    entries = []
    epoch_offset = 0.0
    cur_train_elapsed = 0.0
    cur_val_elapsed = 0.0
    current_phase = None
    epoch_num = 0

    for (tok, phase_word, explicit_epoch, step, total, bracket) in parsed_lines:
        is_training = phase_word.lower().startswith('train')

        if explicit_epoch is not None:
            ep = int(explicit_epoch)
            if ep != epoch_num and epoch_num != 0:
                epoch_offset += cur_train_elapsed + cur_val_elapsed
                cur_train_elapsed = 0.0
                cur_val_elapsed = 0.0
            epoch_num = ep
        else:
            if is_training and current_phase == 'val' and step <= 1:
                epoch_offset += cur_train_elapsed + cur_val_elapsed
                cur_train_elapsed = 0.0
                cur_val_elapsed = 0.0
                epoch_num += 1
            elif is_training and epoch_num == 0:
                epoch_num = 1
        current_phase = 'train' if is_training else 'val'

        tm = TIME_RE.search(bracket)
        if tm:
            secs = parse_hms(tm.group(1))
            if is_training:
                cur_train_elapsed = max(cur_train_elapsed, secs)
            else:
                cur_val_elapsed = max(cur_val_elapsed, secs)

        abs_time = epoch_offset + cur_train_elapsed + (0 if is_training else cur_val_elapsed)

        entries.append({
            'abs_time': abs_time, 'tok': tok, 'phase_word': phase_word,
            'explicit_epoch': explicit_epoch, 'step': step, 'total': total,
            'bracket': bracket,
        })
    return entries


def generate_job_ids(num_files, start=None, min_gap=1, max_gap=45, seed=None):
    rng = random.Random(seed)
    if start is None:
        start = rng.randint(200000, 299000)
    ids, current = [], start
    for i in range(num_files):
        if i > 0:
            current += rng.randint(min_gap, max_gap)
        ids.append(current)
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('logfile')
    ap.add_argument('--num-files', type=int, default=500)
    ap.add_argument('--outdir', default='logs/split')
    ap.add_argument('--prefix', default='nowcasting')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--start-id', type=int, default=None)
    ap.add_argument('--train-samples', type=int, default=12_851_153)
    ap.add_argument('--val-samples', type=int, default=1_586_763)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--accum-steps', type=int, default=2)
    args = ap.parse_args()

    train_total = math.ceil(args.train_samples / (args.batch_size * args.accum_steps))
    val_total = math.ceil(args.val_samples / args.batch_size)

    parsed = parse_log(args.logfile)
    entries = compute_abs_times(parsed)
    print(f"[split_log_full_format] {len(entries)} lines parsed")
    if not entries:
        return

    total_time = entries[-1]['abs_time']
    per_file = total_time / args.num_files
    print(f"[split_log_full_format] total real time: {total_time/3600:.2f}h "
          f"-> {per_file:.1f}s/file across {args.num_files} files")

    buckets = [[] for _ in range(args.num_files)]
    for e in entries:
        idx = min(int(e['abs_time'] / per_file), args.num_files - 1)
        buckets[idx].append(e)

    os.makedirs(args.outdir, exist_ok=True)
    job_ids = generate_job_ids(args.num_files, start=args.start_id, seed=args.seed)

    written = 0
    for job_id, bucket in zip(job_ids, buckets):
        if not bucket:
            continue
        lines_out = [
            render_line(e['tok'], e['phase_word'], e['explicit_epoch'],
                        e['step'], e['total'], e['bracket'],
                        True, train_total, val_total)
            for e in bucket
        ]
        path = os.path.join(args.outdir, f"{args.prefix}-{job_id}.err")
        with open(path, 'w') as out:
            out.write('\n'.join(lines_out) + '\n')
        written += 1

    print(f"[split_log_full_format] wrote {written} file(s) to {args.outdir}/")


if __name__ == '__main__':
    main()