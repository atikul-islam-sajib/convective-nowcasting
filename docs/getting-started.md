# Getting Started

## Requirements

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

## Repository layout

```
config/       # YAML configuration files (data paths, temporal/spatial settings)
src/          # Core library code: acquisition, preprocessing, models, utils
train/        # Training and inference scripts (multimodal, radar, pySTEPS)
slurms/       # Cluster job submission scripts
unittest/     # Sanity checks and parameter-count tests
visualization/# Plotting utilities for radar/satellite fields and predictions
```

## Configuration

All data paths and preprocessing settings are controlled from
[`config/config.yml`](data/config.md). Update `paths.satellite_root` and
`paths.radar_root` to point at your local data before running any script.

## Typical workflow

1. **Preprocess** — generate patch-level metadata (see
   [Data Pipeline](data/preprocessing.md)).
2. **Train** — run one of the scripts in `train/multimodal/` or
   `train/radar/` for the architecture you want.
3. **Infer** — use the corresponding `inference.py` to generate predictions
   and compute metrics.
