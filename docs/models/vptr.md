# VPTR

VPTR [55] combines a convolutional encoder with a Transformer decoder. The input
sequence is first passed through the convolutional encoder, which extracts
spatial features from the radar and satellite observations. These features
are then provided to the Transformer decoder, where the attention
mechanism processes relationships within the encoded sequence. The decoder
uses this information to generate the future frames. In this way, the
convolutional encoder handles the extraction of local spatial features,
while the Transformer is used to model relationships across the sequence.
This combination is suitable for convective precipitation nowcasting
because the prediction requires both the detailed spatial structure of
convective cells and information about how these structures change over
time.

## Simplified flow

```mermaid
flowchart LR
    A["Input sequence"] --> B["Convolutional\nencoder (ResNet)"]
    B --> C["Transformer\ncore (attention)"]
    C --> D["Convolutional\ndecoder (ResNet)"]
    D --> E["Predicted frames"]
```

## Key files (`src/models/vptr/`)

| File | Role |
|---|---|
| `model/VPTR_modules.py` | `VPTREnc`, `VPTRDec`, `VPTRFormerNAR`, `VPTRDisc`. |
| `model/VidHRFormer.py` | Underlying Transformer encoder/decoder blocks. |
| `utils/position_encoding.py` | 1D/2D/3D positional embeddings. |
