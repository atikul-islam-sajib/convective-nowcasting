# Inference

Each configuration (`train/multimodal/inference.py`, `train/radar/inference.py`)
loads all five trained checkpoints and produces predictions for every
evaluated sample.

## Patch-based inference

For inputs larger than the training patch size, predictions are stitched
together from overlapping patches using a cosine blending window:

| Parameter | Value |
|---|---|
| Patch size | 256 |
| Patch overlap | 64 |
| Blend | cosine |

## Ensemble

The ensemble prediction is the unweighted average of the five models'
predictions, computed independently at each lead time:

$$
\hat{y}_{\text{ensemble}} = \frac{1}{5} \sum_{i=1}^{5} \hat{y}_i
$$

## Evaluation metrics

| Metric | Notes |
|---|---|
| MAE, MSE, RMSE, PSNR, SSIM | Continuous metrics |
| CSI, ETS | Categorical metrics at 5 and 15 mm/h thresholds |

Detailed and pooled-mean results are written to CSV files under the
configured `out_dir`.
