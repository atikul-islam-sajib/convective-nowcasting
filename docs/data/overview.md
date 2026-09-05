# Data Pipeline Overview

## Sources

| Source | Product | Resolution | Cadence |
|---|---|---|---|
| Radar | RADOLAN YW (5-minute rainfall accumulation) | 1100 × 900 grid | 5 min |
| Satellite | SEVIRI, channels CH7 and CH9 | Native satellite grid, regridded to radar grid | 5 min |

## Regridding

Radar and satellite data live on different native grids. Satellite pixels are
reprojected using their geographic coordinates onto the fixed radar grid via
linear interpolation (nearest-neighbour as a fallback), so that pixel `(i, j)`
in both modalities corresponds to the same geographic location. See
[Preprocessing](preprocessing.md) for details.

## Input / output sequence

- **Input**: 4 consecutive frames sampled at 5-minute intervals.
- **Output**: predicted radar field at 4 lead times — t+15, t+30, t+45, t+60
  minutes.
- **Spatial size**: resized to 256 × 256 for training.

| Configuration | Input shape | Output shape |
|---|---|---|
| Multimodal (radar + CH7 + CH9) | `(4, 3, 256, 256)` | `(4, 256, 256)` |
| Radar-only | `(4, 1, 256, 256)` | `(4, 256, 256)` |

## Splits

| Split | Period |
|---|---|
| Train | 2015-01-01 – 2022-12-31 |
| Validation | 2023-01-01 – 2023-12-31 |
| Test | 2024-01-01 – 2024-12-31 |

## Seasonal subsetting

Where a "summer-only" subset is used, it is defined as **June–September
(JJAS)**, i.e. months `{6, 7, 8, 9}`, applied consistently across all
preprocessing scripts.
