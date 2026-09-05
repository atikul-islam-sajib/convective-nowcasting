# VPTR

**Source:** `src/models/vptr/`

VPTR combines a convolutional (ResNet) encoder/decoder with a
non-autoregressive Transformer core. The encoder extracts spatial features
from each input frame independently; the Transformer then models
relationships across the encoded sequence before the decoder reconstructs
the predicted frames.

## Structure

- `VPTREnc` / `VPTRDec`: ResNet-based autoencoder (3 downsampling stages).
- `VPTRFormerNAR`: non-autoregressive Transformer core using local window
  attention with relative position encoding.
- A `VPTRDisc` (PatchGAN discriminator) is available for adversarial
  training variants, though the main comparison uses the non-adversarial
  configuration.

!!! note
    As with EarthFormer, this implementation uses a reduced Transformer
    configuration (smaller `d_model`, fewer heads and layers) relative to
    the original paper's larger-scale video prediction benchmarks.

## Key files

| File | Role |
|---|---|
| `model/VPTR_modules.py` | `VPTREnc`, `VPTRDec`, `VPTRFormerNAR`, `VPTRDisc`. |
| `model/VidHRFormer.py` | Underlying Transformer encoder/decoder blocks. |
| `utils/position_encoding.py` | 1D/2D/3D positional embeddings. |
