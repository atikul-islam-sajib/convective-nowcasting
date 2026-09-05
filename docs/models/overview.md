# Models Overview

## Sequence-to-sequence formulation

The forecasting problem is posed as a supervised sequence-to-sequence task:

$$
f_\theta : \mathcal{X} \rightarrow \mathcal{Y} \qquad (4.1)
$$

where $\mathcal{X}$ is the input sequence of (multimodal or radar-only)
observations, $\mathcal{Y}$ is the corresponding sequence of future radar
precipitation fields, and $f_\theta$ is the forecasting model parameterized
by $\theta$. Radar precipitation is the prediction target in both input
configurations, since it is the operational reference for nowcasting.

## Input fusion strategy

Radar and satellite channels are combined via simple channel concatenation
(no extra parameters added):

$$
\underbrace{(256{\times}256{\times}1)}_{\text{Radar}} +
\underbrace{(256{\times}256{\times}2)}_{\text{CH7, CH9}}
\xrightarrow{\text{concatenate}}
\underbrace{(256{\times}256{\times}3)}_{\text{Input frame}}
$$

$$
\underbrace{4 \times (256{\times}256{\times}3)}_{\text{4 timesteps}}
\xrightarrow{\text{stack in time}}
\underbrace{4 \times 256 \times 256 \times 3}_{\text{Model input}}
$$

## Architectures

The architectures evaluated in this study include recurrent, convolutional,
and Transformer-based models, along with different strategies for
representing temporal information. Each is trained separately on the
multimodal and radar-only input configurations.

| Model | Core mechanism | Multimodal input | Radar-only input | Output |
|---|---|---|---|---|
| [ConvLSTM](convlstm.md) | Convolutional LSTM (recurrent encoder-forecaster) | `(1,12,256,256)` | `(1,4,256,256)` | `(1,4,256,256)` |
| [SimVP](simvp.md) | Pure CNN (encoder-translator-decoder, no recurrence/attention) | `(1,12,256,256)` | `(1,4,256,256)` | `(1,4,256,256)` |
| [SmaAt-UNet](smaat-unet.md) | CNN-based U-Net with CBAM attention | `(1,12,256,256)` | `(1,4,256,256)` | `(1,4,256,256)` |
| [EarthFormer](earthformer.md) | Cuboid Transformer with local windowed attention | `(1,12,256,256)` | `(1,4,256,256)` | `(1,4,256,256)` |
| [VPTR](vptr.md) | ResNet autoencoder + non-autoregressive Transformer core | `(1,12,256,256)` | `(1,4,256,256)` | `(1,4,256,256)` |

!!! note
    12 channels = 4 input frames × 3 channels (radar + CH7 + CH9).
    4 channels (radar-only) = 4 input frames × 1 channel.
    4 output channels = 4 forecast horizons (t+15, t+30, t+45, t+60).

## Ensemble Model

In addition to the five models, an ensemble modeling technique has been
incorporated by considering predictions of all five models [56]. The ensemble
prediction for each forecast horizon is the arithmetic mean of the five
model predictions:

$$
\hat{y}_{\text{ensemble}} = \frac{1}{5}\sum_{i=1}^{5} \hat{y}_i \qquad (4.2)
$$

where $\hat{y}_i$ is the prediction of model $i$. Taking the mean of all
five models may reduce the influence of errors associated with individual
architectures, depending on the degree of diversity between their
predictions — see [Results & Findings](../results.md) for how well this
held up empirically (H3).
