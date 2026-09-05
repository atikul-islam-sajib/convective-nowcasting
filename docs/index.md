# Convective Nowcasting

This repository implements and compares five deep learning architectures for
short-term (0–60 minute) convective precipitation nowcasting, using paired
RADOLAN YW radar and SEVIRI (CH7, CH9) satellite observations over Germany.

## What this project does

- Evaluates **five architectures** — ConvLSTM, SimVP, SmaAt-UNet, EarthFormer,
  and VPTR — representing recurrent, convolutional, and Transformer-based
  design principles.
- Compares **multimodal** (radar + satellite) input against a **radar-only**
  configuration to test whether satellite fusion improves nowcasting skill.
- Evaluates an **ensemble** formed from the unweighted average of all five
  models' predictions.
- Benchmarks all deep learning models against a classical **pySTEPS**
  extrapolation baseline.

## Quick facts

| Property | Value |
|---|---|
| Input window | 4 frames, 5-minute cadence |
| Forecast horizons | t+15, t+30, t+45, t+60 min |
| Spatial resolution | 256 × 256 |
| Multimodal input channels | 12 (4 frames × [radar, CH7, CH9]) |
| Radar-only input channels | 4 (4 frames × [radar]) |
| Train / Val / Test years | 2015–2022 / 2023 / 2024 |

See [Getting Started](getting-started.md) to set up the environment, or jump
straight to the [Models overview](models/overview.md) or
[Data Pipeline](data/overview.md).
