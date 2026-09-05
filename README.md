# Convective Precipitation Nowcasting

Convective precipitation is highly localized and evolves continuously over
short timescales, making short-term precipitation forecasting particularly
challenging despite its importance for early warning and weather-sensitive
operations. This repository evaluates five deep learning architectures for
convective precipitation nowcasting over Germany and investigates whether
combining radar and satellite observations improves forecasting performance
compared with a radar-only configuration. Models were trained using paired
RADOLAN radar and SEVIRI satellite observations collected between 2015 and
2024, with an ensemble of all five architectures also evaluated. No single
architecture performed best across all metrics: VPTR achieved the strongest
individual performance overall, particularly at longer lead times, while
the ensemble was competitive at shorter horizons but did not consistently
outperform the best individual model. Integrating satellite observations
generally improved forecast quality for the deep learning models, though
this benefit did not extend to the extrapolation-based baseline.

**Full documentation:** https://atikul-islam-sajib.github.io/convective-nowcasting/

## Overview

Two input modes are supported end-to-end (data loading, training, and
inference), sharing the same dataset class, loss functions, and evaluation
metrics — only the input channel count differs:

| Mode | Input | Scripts |
|---|---|---|
| **Multimodal** | Radar + SEVIRI CH7/CH9, 4 frames (12 channels) | `train/multimodal/` |
| **Radar-only** | Radar only, 4 frames (4 channels) | `train/radar/` |

Five architectures are implemented under `src/models/`, each with a
matching training script in both configurations: **ConvLSTM**, **SimVP**,
**SmaAt-UNet**, **EarthFormer**, **VPTR**. A classical **pySTEPS**
extrapolation baseline is included for comparison against a non-deep-learning
approach.

## Input / output sequence

At each reference time $t$, the input sequence consists of 4 consecutive
frames, each covering a 5-minute observation period:

$$
\mathcal{X}_t = \left\\{ x_t^{(1)}, x_t^{(2)}, x_t^{(3)}, x_t^{(4)} \right\\}
$$

The model predicts the radar precipitation field at 4 future lead times:

$$
\mathcal{Y} = \left\\{ y_{t+15},\; y_{t+30},\; y_{t+45},\; y_{t+60} \right\\}
$$

| Configuration | Input shape | Output shape |
|---|---|---|
| Multimodal (radar + CH7 + CH9) | `(4, 3, 256, 256)` | `(4, 256, 256)` |
| Radar-only | `(4, 1, 256, 256)` | `(4, 256, 256)` |

## Loss function

All models are trained with a hybrid loss combining an intensity-weighted
MAE, a gradient sharpness penalty, and horizon-dependent weighting:

$$
\mathcal{L}_h = \frac{1}{|\mathcal{M}|} \sum_{(x,y)\in\mathcal{M}}
w(x,y)\,\big|\hat{y}_{x,y,h} - y_{x,y,h}\big| + \lambda\,\mathcal{L}_{\text{grad},h}
$$

$$
\mathcal{L}_{\text{total}} = \sum_{h=1}^{4} \bar{w}_h\,\mathcal{L}_h
$$

where $w(x,y)$ up-weights heavier rainfall intensities, $\mathcal{L}_{\text{grad},h}$
penalizes over-smoothed predictions ($\lambda = 0.2$), and $\bar{w}_h$
emphasizes longer lead times.

## Evaluation metrics

Models are compared on the held-out test set using five metrics — MAE and
MSE (lower is better), CSI, ETS, and PSNR (higher is better):

$$
\text{CSI} = \frac{H}{H+M+FA}, \qquad
\text{ETS} = \frac{H-H_c}{H+M+FA-H_c}, \qquad
H_c = \frac{(H+M)(H+FA)}{N}
$$

where $H$, $M$, $FA$ are hits, misses, and false alarms at a given
precipitation threshold, and $H_c$ is the number of hits expected by
chance. **ETS is used as the model-selection criterion** on the validation
set, since it corrects for chance agreement on this class-imbalanced data.
Full derivations for MAE, MSE, and PSNR are in the
[documentation](https://atikul-islam-sajib.github.io/convective-nowcasting/).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Data preparation

Raw inputs are per-timestep `.npy` radar frames and per-channel satellite
frames. Before training, the following preprocessing steps are applied
(see `src/preprocess/`):

1. **Regridding** — satellite pixels are reprojected via their geographic
   coordinates onto the fixed 1100x900 radar grid using bilinear
   interpolation (nearest-neighbour fallback), so radar and satellite
   pixels correspond to the same location.
2. **Normalization** — radar values are log-transformed and clipped at
   128 mm/h; satellite channels are Z-score normalized using
   training-set statistics.
3. **Patch extraction** — 256x256 patches (stride 64) are extracted from
   the full radar/satellite grid.
4. **Two-bucket quantile sampling** — patches are split into a "heavy
   rain" bucket (top 40% by 99th-percentile intensity) and the rest,
   then resampled to roughly balance the two, addressing the natural
   imbalance between dry and precipitating pixels.
5. **Metadata generation** — sample windows (4 input frames, 4 forecast
   horizons at 15/30/45/60 min) are indexed into CSV files consumed
   directly by the data loaders.

Run the pipeline scripts in `src/preprocess/` in order; see the
[full documentation](https://atikul-islam-sajib.github.io/convective-nowcasting/)
for the exact script sequence and configuration options.

## Configuration

`config/config.yml` holds shared settings: data paths, satellite channels,
temporal history/cadence, forecast horizons, spatial transform, and
train/val/test splits (2015-2022 / 2023 / 2024). Per-model training
hyperparameters live at the top of each `train_*.py` script.

## Training

Both configurations share an identical interface — only the input channel
count and a `radar_only` flag differ.

**Multimodal (radar + satellite):**

```bash
python train/multimodal/train_ConvLSTM.py       # ConvLSTM
python train/multimodal/train_simVP.py          # SimVP
python train/multimodal/train_smaAt_UNet.py     # SmaAt-UNet
python train/multimodal/train_earthformer.py    # EarthFormer
python train/multimodal/train_VPTR.py           # VPTR
```

**Radar-only:**

```bash
python train/radar/train_ConvLSTM.py            # ConvLSTM
python train/radar/train_simVP.py               # SimVP
python train/radar/train_smaAt_UNet.py          # SmaAt-UNet
python train/radar/train_earthformer.py         # EarthFormer
python train/radar/train_VPTR.py                # VPTR
```

### pySTEPS baseline

A classical, non-deep-learning extrapolation baseline using Lucas-Kanade
optical flow, evaluated on the same radar-only and multimodal inputs and
lead times as the deep learning models. Unlike the deep learning models,
pySTEPS requires no parameter training — it estimates a motion field from
recent radar frames and extrapolates it forward:

```bash
python train/pySTEPS/train_baseline.py
```

## Inference

Full-image (patch-stitched) multi-model comparison and ensembling:

```bash
python train/multimodal/inference.py --num_samples 10
python train/radar/inference.py      --num_samples 10
```

Each run writes per-sample comparison figures plus metric CSVs to the
configured output directory.

## Repository layout

```
convective-nowcasting/
├── config/          # YAML configs (data paths, model hyperparameters)
├── src/
│   ├── datasets/    # PyTorch Dataset classes
│   ├── models/      # ConvLSTM, SimVP, SmaAt-UNet, EarthFormer, VPTR
│   ├── preprocess/  # Metadata / statistics generation scripts
│   └── utils/       # Config loading, transforms, IO
├── train/
│   ├── multimodal/  # Satellite + radar training and inference
│   ├── radar/       # Radar-only training and inference
│   └── pySTEPS/     # Classical baseline
├── unittest/        # Sanity checks
└── requirements.txt
```

## Author

**Atikul Islam Sajib**

Supervisors: Prof. Dr. Stefan Edlich, Dr. Noelia Otero Felipe

## License

See [LICENSE](LICENSE).
