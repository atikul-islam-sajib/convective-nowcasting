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

All five architectures are trained under identical optimization settings
(Table 4.3 in the thesis):

| Setting | Value |
|---|---|
| Optimizer | AdamW ($\beta_1=0.9$, $\beta_2=0.999$) |
| Learning rate | 1e-4 |
| Weight decay | 1e-3 |
| Gradient clipping | 0.5 |
| LR schedule | One-cycle (max LR 3e-4) |
| Batch size | 16 |
| Max epochs | 50 |
| Early stopping patience | 30 epochs |
| Mixed precision | Yes |
| Data workers | 24 (prefetch factor 4) |
| Hardware | 4 × NVIDIA Tesla V100-PCIE (32 GB) |
| Framework | PyTorch |
| Containerization | Docker |
| Experiment tracking | MLflow |
| Hyperparameter search | Optuna |

For each architecture, the checkpoint with the highest **ETS** on the
validation set was selected — ETS accounts for hits expected by chance and
is well suited to the class-imbalanced precipitation data used here.

See [Loss Function](loss.md) for the training objective.
