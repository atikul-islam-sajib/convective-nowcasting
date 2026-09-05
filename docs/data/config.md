# Configuration Reference

All data and preprocessing settings live in `config/config.yml`. This page
summarizes the sections; see the file itself for the authoritative values.

```yaml
paths:
  satellite_root: <path to SEVIRI data>
  radar_root: <path to RADOLAN YW data>

satellite:
  channels: [7, 9]
  cadence_minutes: 5

temporal:
  history_minutes: 20     # 4 input frames x 5 min
  radar_lead_minutes: 60  # 4 forecast horizons x 15 min

spatial:
  mode: resize
  height: 256
  width: 256

transform:
  satellite:
    normalization: zscore
  radar:
    log_transform: true
    clip_max: 128.0

patch:
  height: 256
  width: 256
  stride: 64

splits:
  train: {start: 2015-01-01, end: 2022-12-31}
  val:   {start: 2023-01-01, end: 2023-12-31}
  test:  {start: 2024-01-01, end: 2024-12-31}
```

!!! warning
    Some legacy scripts (e.g. `generate_metadata_patch.py`) define their own
    argparse defaults that can drift from `config.yml` over time. Always
    verify a script's actual defaults with `--help` before relying on them.
