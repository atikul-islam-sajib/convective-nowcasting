# EarthFormer

**Source:** `src/models/earthformer/`

EarthFormer processes spatiotemporal data using a Cuboid Transformer: the
input sequence is divided into local 3D cuboids spanning both space and time,
and self-/cross-attention is applied within and across these cuboids so that
information at different locations and time steps can be related.

## Structure

- Initial convolutional downsampling stage.
- Multi-stage cuboid encoder and decoder using local windowed attention.
- Non-autoregressive decoder: the initial query is derived from the
  encoder's memory via interpolation, rather than generated frame by frame.

!!! note
    This implementation uses a reduced configuration (smaller channel width
    and fewer attention layers) relative to the original paper's SEVIR
    benchmark setup, in order to keep training computationally tractable.

## Key files

| File | Role |
|---|---|
| `cuboid_transformer/cuboid_transformer.py` | `CuboidTransformerModel`, the main architecture. |
| `cuboid_transformer/cuboid_transformer_patterns.py` | Registry of attention patterns (axial, cross-K×K, etc.). |
| `config.py` | Path configuration (not architecture hyperparameters). |
