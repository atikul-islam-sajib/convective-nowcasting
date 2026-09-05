# Data Pipeline Overview

## Sources

| Source | Product | Native resolution | Cadence |
|---|---|---|---|
| Radar | RADOLAN YW (5-minute rainfall accumulation) | 1100 × 900 grid | 5 min |
| Satellite | SEVIRI (MSG), channels CH7 (8.7 µm) and CH9 (10.8 µm) | 175 × 320, regridded to radar grid | 5 min |

Channel 7 provides cloud-property information; Channel 9 provides cloud-top
temperature, both relevant to characterizing convective cloud systems. Two
of SEVIRI's 12 available spectral channels are used.

## Dataset scale (2015–2024, 3,653 days)

| | Radar | Satellite (CH7 + CH9) |
|---|---|---|
| Images per day | 288 | 288 per channel |
| Total images | ~1,052,064 | ~2,104,128 |

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
