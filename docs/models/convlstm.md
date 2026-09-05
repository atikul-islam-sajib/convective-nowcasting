# ConvLSTM

**Source:** `src/models/convlstm/`

ConvLSTM combines convolution with the LSTM recurrent structure. The model
processes the input sequence one frame at a time; convolution extracts
spatial features at each step, while the hidden and cell states carry
information forward through time. This allows the model to track the recent
movement and evolution of precipitation systems across the input sequence.

## Structure

- 3 encoder stages, each pairing a convolutional downsampling layer with a
  ConvLSTM cell.
- 3 mirrored decoder stages that upsample back to the original resolution.
- Final output layer has **no activation function**, consistent with a
  regression task.

## Key files

| File | Role |
|---|---|
| `model.py` | Generic Encoder-Decoder (`ED`) wrapper. |
| `encoder.py` / `decoder.py` | Stage-wise encoder/decoder implementations. |
| `ConvRNN.py` | The ConvLSTM cell implementation. |
| `net_params.py` / `net_params_multihorizon.py` / `net_params_radar.py` | Per-configuration layer parameters. |
