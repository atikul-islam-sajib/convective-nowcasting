# Preprocessing

## Key scripts (`src/preprocess/`)

| Script | Purpose |
|---|---|
| `compute_mean_std_satellite.py` | Computes per-channel normalization statistics for satellite data. |
| `generate_metadata_patch.py` | Generates patch-level metadata (valid sample windows, NaN filtering) for training. |
| `sampling_patch_metadata.py` | Applies importance sampling, over-representing high-precipitation ("top bucket") patches. |
| `compute_presample_thresholds.py` | Computes the percentile threshold used to define the "top bucket" for sampling. |
| `generate_test_data_full_image.py` | Generates full-image (non-patched) test samples for final evaluation. |
| `generate_metadata_multihorizon_season.py` | Generates metadata with an explicit seasonal (JJAS) filter. |

## Patch configuration

| Parameter | Value |
|---|---|
| Patch size | 256 × 256 |
| Patch stride | 64 |
| NaN rejection threshold | 0.3 (patches with ≥30% NaN coverage are discarded) |

## Importance sampling

To address the natural imbalance between light and heavy precipitation,
patches are sampled with two parameters:

- **`r` (top-bucket ratio)** — the fraction of patches (ranked by the 99th
  percentile rainfall in the sequence) considered "heavy rain": **r = 0.40**.
- **`p` (keep probability)** — the probability that a top-bucket patch is
  retained; the remaining patches are kept with probability `1 - p`:
  **p = 0.60**.

## Normalization

| Field | Transform |
|---|---|
| Radar | Log transform, clipped at 128 mm/h |
| Satellite | Z-score normalization (per-channel mean/std) |

!!! note
    Config values above should always be cross-checked against the current
    [`config/config.yml`](config.md), which is the single source of truth
    used by all training scripts.
