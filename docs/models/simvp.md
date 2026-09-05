# SimVP

**Source:** `src/models/simvp/`

SimVP uses a purely convolutional encoder-translator-decoder design, with
**no recurrent connections and no attention mechanism**. This makes it a
useful comparison point against the recurrent (ConvLSTM) and Transformer-based
(EarthFormer, VPTR) architectures.

## Structure

- **Encoder**: spatial downsampling via stacked `ConvSC` blocks.
- **Translator (`Mid_Xnet`)**: Inception-style blocks using multiple parallel
  kernel sizes (3, 5, 7, 11) to model temporal evolution in the latent space.
- **Decoder**: mirrors the encoder, upsampling back to the input resolution.

!!! note
    The "Inception" module here is a small, custom, from-scratch block
    inspired by the multi-kernel idea in GoogLeNet's Inception module — it is
    **not** a pretrained ImageNet Inception network.

## Key files

| File | Role |
|---|---|
| `model.py` | `Encoder`, `Decoder`, `Mid_Xnet`, and the top-level `SimVP` module. |
| `modules.py` | `ConvSC` and `Inception` building blocks. |
