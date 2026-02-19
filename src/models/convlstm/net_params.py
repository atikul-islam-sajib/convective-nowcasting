# src/models/convlstm/utils.py
# UPDATED FOR 3 INPUT CHANNELS (2 satellite + 1 radar per timestep)

from collections import OrderedDict
from .ConvRNN import CLSTM_cell

# ============================================================
# ENCODER: 3 channels → 64 → 96 → 96
# ============================================================
convlstm_encoder_params = [
    [
        # Stage 1: 3 input channels → 16 features
        OrderedDict({'conv1_leaky_1': [3, 16, 3, 1, 1]}),  # ← Changed from 2 to 3
        # Stage 2: 64 → 64 (downsample)
        OrderedDict({'conv2_leaky_1': [64, 64, 3, 2, 1]}),
        # Stage 3: 96 → 96 (downsample)
        OrderedDict({'conv3_leaky_1': [96, 96, 3, 2, 1]}),
    ],
    [
        # ConvLSTM cells for temporal processing
        CLSTM_cell(shape=(256, 256), input_channels=16, filter_size=5, num_features=64),  # ← Updated shape to 256x256
        CLSTM_cell(shape=(128, 128), input_channels=64, filter_size=5, num_features=96),  # ← Updated shape to 128x128
        CLSTM_cell(shape=(64, 64), input_channels=96, filter_size=5, num_features=96)     # ← Updated shape to 64x64
    ]
]

# ============================================================
# DECODER: 96 → 96 → 64 → 1
# ============================================================
convlstm_decoder_params = [
    [
        # Stage 1: Upsample 96 → 96
        OrderedDict({'deconv1_leaky_1': [96, 96, 4, 2, 1]}),
        # Stage 2: Upsample 96 → 96
        OrderedDict({'deconv2_leaky_1': [96, 96, 4, 2, 1]}),
        # Stage 3: Final conv 64 → 16 → 1 (output)
        # CRITICAL: Last layer has NO activation (regression task!)
        OrderedDict({
            'conv3_leaky_1': [64, 16, 3, 1, 1],  # Has LeakyReLU
            'conv4_final': [16, 1, 1, 1, 0]      # NO activation (pure regression output)
        }),
    ],
    [
        # ConvLSTM cells for decoder
        CLSTM_cell(shape=(64, 64), input_channels=96, filter_size=5, num_features=96),    # ← Updated shape to 64x64
        CLSTM_cell(shape=(128, 128), input_channels=96, filter_size=5, num_features=96),  # ← Updated shape to 128x128
        CLSTM_cell(shape=(256, 256), input_channels=96, filter_size=5, num_features=64),  # ← Updated shape to 256x256
    ]
]
