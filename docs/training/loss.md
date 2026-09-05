# Loss Function

All five architectures are trained with the same **hybrid** loss
(`loss_type = 'hybrid_weighted_mae'`), combining three components.

## 1. Five-tier precipitation-weighted MAE

Applies increasing weight to heavier, more hazardous rainfall intensities:

| Threshold (mm/h) | Weight multiplier |
|---|---|
| 3 | 3× |
| 7 | 7× |
| 15 | 15× |
| 25 | 25× |
| 35 | 35× |

Implemented in `HybridWeightedMAE` (see `train/*/train_<Model>.py`).

## 2. Gradient sharpness penalty

A finite-difference gradient loss (`GradientSharpnessLoss`) discourages
overly smooth predictions by penalizing differences between the predicted
and ground-truth spatial gradients.

| Parameter | Value |
|---|---|
| `gradient_loss_weight` | 0.2 |

## 3. Horizon-dependent weighting

Longer lead times are weighted more heavily, since they are both harder to
predict and more valuable operationally:

| Lead time | Weight |
|---|---|
| t+15 | 1.0 |
| t+30 | 1.5 |
| t+45 | 2.0 |
| t+60 | 2.5 |

!!! note
    These values are the ones stored in each script's `Config` class and
    actually passed to the loss constructors. Some `__init__` default
    arguments and print statements in earlier revisions did not match these
    values; the `Config` class is the source of truth.
