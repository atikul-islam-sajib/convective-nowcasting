# SmaAt-UNet

**Source:** `src/models/smaat_unet/`

SmaAt-UNet follows a U-Net encoder-decoder structure, using depthwise
separable convolutions to reduce parameter count and CBAM (Convolutional
Block Attention Module) gates in the skip connections to emphasize
informative regions — useful here since convective cells can occupy only a
small fraction of the radar field.

## Structure

- 5 encoder stages (`inc` + 4 `DownDS` stages), each followed by a CBAM block.
- 4 decoder stages (`UpDS`), using skip connections from the corresponding
  encoder stage.
- Channel progression: 64 → 128 → 256 → 512 → 512 (with `bilinear=True`).

## Key files

| File | Role |
|---|---|
| `SmaAt_UNet.py` | Top-level model definition. |
| `unet_parts_depthwise_separable.py` | `DoubleConvDS`, `DownDS`, `UpDS`, `OutConv`. |
| `layers.py` | CBAM attention module. |
