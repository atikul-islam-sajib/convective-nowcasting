# Data Pipeline Overview

## Sources

| Source | Product | Native resolution | Cadence |
|---|---|---|---|
| Radar | RADOLAN YW (5-minute rainfall accumulation) | 1100 × 900 grid | 5 min |
| Satellite | SEVIRI (MSG), channels CH7 (8.7 µm) and CH9 (10.8 µm) [37] | 175 × 320, regridded to radar grid | 5 min |

Channel 7 provides cloud-property information; Channel 9 provides cloud-top
temperature, both relevant to characterizing convective cloud systems. Two
of SEVIRI's 12 available spectral channels are used.

## Dataset scale (2015–2024, 3,653 days)

| | Radar | Satellite (CH7 + CH9) |
|---|---|---|
| Images per day | 288 | 288 per channel |

## Regridding

Radar and satellite data live on different native grids. Satellite pixels are
reprojected using their geographic coordinates onto the fixed radar grid via
linear interpolation (nearest-neighbour as a fallback), so that pixel `(i, j)`
in both modalities corresponds to the same geographic location. See
[Preprocessing](preprocessing.md) for details.

## Input / output sequence

At reference time $t$, the input sequence is:

$$
\mathcal{X}_t = \left\{ x_t^{(1)}, x_t^{(2)}, x_t^{(3)}, x_t^{(4)} \right\}
\qquad (3.9)
$$

where $x_t^{(i)}$ is the $i$-th frame, each covering a distinct 5-minute
observation period. Four such frames jointly represent a 20-minute
observation window ending at $t$.

**Output**: predicted radar field at 4 lead times — t+15, t+30, t+45, t+60
minutes. **Spatial size**: resized to 256 × 256 for training.

### Table 3.8 — Input and output sequence configuration

| Property | Input | Output | Value |
|---|---|---|---|
| Frames | $N_{in}=4$ | $N_{out}=4$ | 4 |
| Frame length | 5 min | — | — |
| Lead times | — | 15, 30, 45, 60 min | — |
| Window | 20 min | 60 min | — |
| Channels | Radar + CH7 + CH9 | Radar only | 3 / 1 |
| Spatial size | 256 × 256 pixels | | |

| Configuration | Input shape | Output shape |
|---|---|---|
| Multimodal (radar + CH7 + CH9) | `(4, 3, 256, 256)` | `(4, 256, 256)` |
| Radar-only | `(4, 1, 256, 256)` | `(4, 256, 256)` |

## Splits

### Table 3.9 — Temporal dataset split across periods

| Split | Period | Duration | Purpose |
|---|---|---|---|
| Training | 2015–2022 | 8 years | Parameter optimisation |
| Validation | 2023 | 1 year | Hyperparameter tuning |
| Test | 2024 | 1 year | Final evaluation |

## Seasonal subsetting

Where a "summer-only" subset is used, it is defined as **June–September
(JJAS)**, i.e. months `{6, 7, 8, 9}`, applied consistently across all
preprocessing scripts.

See [Dataset Statistics](statistics.md) for the full year-by-year breakdown
(availability, NaN rates, rainfall intensity, thresholds, seasonality) that
motivated these preprocessing choices.
