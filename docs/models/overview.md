# Models Overview

Five architectures are evaluated, representing recurrent, convolutional, and
Transformer-based design principles for spatiotemporal forecasting. Each is
trained separately on the multimodal and radar-only input configurations.

| Model | Core mechanism | Multimodal input | Radar-only input | Output |
|---|---|---|---|---|
| [ConvLSTM](convlstm.md) | Convolutional LSTM (recurrent encoder-forecaster) | `(1,12,256,256)` | `(1,4,256,256)` | `(1,4,256,256)` |
| [SimVP](simvp.md) | Pure CNN (encoder-translator-decoder, no recurrence/attention) | `(1,12,256,256)` | `(1,4,256,256)` | `(1,4,256,256)` |
| [SmaAt-UNet](smaat-unet.md) | CNN-based U-Net with CBAM attention | `(1,12,256,256)` | `(1,4,256,256)` | `(1,4,256,256)` |
| [EarthFormer](earthformer.md) | Cuboid Transformer with local windowed attention | `(1,12,256,256)` | `(1,4,256,256)` | `(1,4,256,256)` |
| [VPTR](vptr.md) | ResNet autoencoder + non-autoregressive Transformer core | `(1,12,256,256)` | `(1,4,256,256)` | `(1,4,256,256)` |

An **ensemble** prediction is also evaluated, formed as the unweighted
average of all five models' outputs at each lead time.

!!! note
    12 channels = 4 input frames × 3 channels (radar + CH7 + CH9).
    4 channels (radar-only) = 4 input frames × 1 channel.
    4 output channels = 4 forecast horizons (t+15, t+30, t+45, t+60).
