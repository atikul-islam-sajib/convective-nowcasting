# Loss Function

All five architectures are trained with the same **hybrid** loss
(`loss_type = 'hybrid_weighted_mae'`), combining three components: an
intensity-weighted MAE, a gradient sharpness penalty, and horizon-dependent
weighting.

## 1. Intensity-weighted MAE (per horizon)

For each forecast horizon $h$:

$$
\mathcal{L}_h = \frac{1}{|\mathcal{M}|} \sum_{(x,y)\in\mathcal{M}}
w(x,y)\,\big|\hat{y}_{x,y,h} - y_{x,y,h}\big| + \lambda\,\mathcal{L}_{\text{grad},h}
\qquad (4.4)
$$

where $\hat{y}_{x,y,h}$ and $y_{x,y,h}$ are the predicted and observed
values at pixel $(x,y)$ for horizon $h$, $w(x,y)$ is the precipitation-based
pixel weight, and $\mathcal{M}$ contains the valid (non-NaN) pixels. The
pixel weight itself, for the single-horizon MAE term, is:

$$
\mathcal{L}_{\text{MAE}} = \frac{1}{|\mathcal{M}|}\sum_{(x,y)\in\mathcal{M}}
w(x,y)\,\big|\hat{y}_{x,y} - y_{x,y}\big| \qquad (4.3)
$$

Thresholds and weights (Table 4.1) were derived directly from the training
data's precipitation distribution:

### Table 4.1 — Intensity-weighted threshold scheme

| Tier | Threshold (mm/h) | Weight |
|---|---|---|
| 1 | ≥ 3.0 | 3.0 |
| 2 | ≥ 7.0 | 7.0 |
| 3 | ≥ 15.0 | 15.0 |
| 4 | ≥ 25.0 | 25.0 |
| 5 | ≥ 35.0 | 35.0 |

## 2. Gradient sharpness penalty

$\mathcal{L}_{\text{grad},h}$ compares the spatial gradients of the
predicted and target fields, encouraging sharper precipitation structures
and reducing excessive smoothing. Its contribution is controlled by
$\lambda$:

| Parameter | Value |
|---|---|
| $\lambda$ (`gradient_loss_weight`) | 0.2 (fixed for all models) |

## 3. Horizon-dependent weighting

The four per-horizon losses are combined using normalized temporal weights:

$$
\mathcal{L}_{\text{total}} = \sum_{h=1}^{4} \bar{w}_h\,\mathcal{L}_h
\qquad (4.5)
$$

where $\bar{w}_h$ is the raw horizon weight divided by the sum of all four
weights. Later horizons receive larger weights to emphasize longer-term
forecast accuracy (Table 4.2):

### Table 4.2 — Temporal horizon weights

| Horizon | Lead time (min) | Weight $w_h$ |
|---|---|---|
| 1 | 15 | 1.0 |
| 2 | 30 | 1.5 |
| 3 | 45 | 2.0 |
| 4 | 60 | 2.5 |

!!! note
    These values are the ones stored in each script's `Config` class and
    actually passed to the loss constructors. Some `__init__` default
    arguments and print statements in earlier revisions did not match these
    values; the `Config` class is the source of truth.
