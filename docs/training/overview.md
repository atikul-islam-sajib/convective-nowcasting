# Training Overview

## Scripts

| Directory | Contents |
|---|---|
| `train/multimodal/` | Training and inference scripts using radar + satellite input. |
| `train/radar/` | Training and inference scripts using radar-only input. |
| `train/pySTEPS/` | Classical extrapolation-based baseline (not a deep learning model). |

Each architecture has a corresponding `train_<Model>.py` script in both the
`multimodal/` and `radar/` directories, plus a shared `inference.py` per
configuration that loads all five trained checkpoints and computes the
ensemble prediction.

## Common training settings

All five architectures are trained under identical optimization settings:

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| One-cycle max LR | 3e-4 |

See [Loss Function](loss.md) for the objective used during training.
