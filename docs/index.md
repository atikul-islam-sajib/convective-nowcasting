# Convective Precipitation Nowcasting Using Deep Learning

**Integration of Satellite and Radar Imagery**

*M.Sc. Data Science thesis — Atikul Islam Sajib, Berliner Hochschule für
Technik Berlin*

| Role | Name |
|---|---|
| Supervisor | Prof. Dr. Stefan Edlich (Berliner Hochschule für Technik) |
| Day-to-day supervisor | Dr. Noelia Otero Felipe (Fraunhofer HHI) |
| Reviewer | Prof. Dr. Sören Werth (Berliner Hochschule für Technik) |

## Abstract

Convective precipitation is highly localized and evolves continuously over
short timescales, making short-term precipitation forecasting particularly
challenging despite its importance for early warning and weather-sensitive
operations. This thesis evaluates five deep learning architectures for
convective precipitation nowcasting over Germany and investigates whether
combining radar and satellite observations improves forecasting performance
compared with a radar-only configuration. The models were trained using
paired RADOLAN radar and SEVIRI satellite observations collected between
2015 and 2024. In addition to the five individual architectures, an ensemble
combining their predictions was also evaluated. Forecast performance was
assessed at lead times of 15, 30, 45, and 60 minutes using both continuous
and categorical evaluation metrics.

No single architecture performed best across all metrics. VPTR achieved the
strongest individual performance overall, particularly at longer lead
times, while the ensemble was competitive at shorter horizons but its
relative ranking declined as lead time increased. Integrating satellite
observations generally improved forecast quality for the deep learning
models, although this benefit did not extend to the extrapolation-based
baseline. Severe convective precipitation remained the most difficult to
predict because such events were underrepresented in the training data.

## Research hypotheses

| | Hypothesis |
|---|---|
| **H1** | The choice of deep learning architecture affects precipitation nowcasting performance under comparable experimental conditions. |
| **H2** | Combining radar and satellite observations provides more accurate precipitation forecasts than using radar observations alone. |
| **H3** | An ensemble of multiple deep learning models improves precipitation nowcasting performance compared with individual architectures. |

See [Results & Findings](results.md) for how each hypothesis was examined.

## Quick facts

| Property | Value |
|---|---|
| Input window | 4 frames, 5-minute cadence |
| Forecast horizons | t+15, t+30, t+45, t+60 min |
| Spatial resolution | 256 × 256 |
| Multimodal input channels | 12 (4 frames × [radar, CH7, CH9]) |
| Radar-only input channels | 4 (4 frames × [radar]) |
| Train / Val / Test years | 2015–2022 / 2023 / 2024 |
| Season | June–September (JJAS) |

See [Getting Started](getting-started.md) to set up the environment, or jump
straight to the [Models overview](models/overview.md) or
[Data Pipeline](data/overview.md).
