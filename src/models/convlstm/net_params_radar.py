# src/models/convlstm/net_params_multihorizon.py
from collections import OrderedDict
from .ConvRNN import CLSTM_cell

# ============================================================
# ENCODER — fixed for 256x256 input, in_channels per timestep
# ============================================================
def get_convlstm_encoder_params(in_channels: int = 3):
    """
    Returns encoder params with dynamic input channel count.
    Default in_channels=3 preserves original satellite+radar behaviour.
    Pass in_channels=1 for radar-only mode.
    """
    return [
        [
            OrderedDict({'conv1_leaky_1': [in_channels, 16, 3, 1, 1]}),
            OrderedDict({'conv2_leaky_1': [64, 64, 3, 2, 1]}),
            OrderedDict({'conv3_leaky_1': [96, 96, 3, 2, 1]}),
        ],
        [
            CLSTM_cell(shape=(256, 256), input_channels=16, filter_size=5, num_features=64),
            CLSTM_cell(shape=(128, 128), input_channels=64, filter_size=5, num_features=96),
            CLSTM_cell(shape=(64,  64),  input_channels=96, filter_size=5, num_features=96),
        ]
    ]

# Keep the original variable as a convenience alias (backward-compatible)
convlstm_encoder_params = get_convlstm_encoder_params(in_channels=3)

# ============================================================
# DECODER — fixed for 256x256 output
# ============================================================
def get_convlstm_decoder_params(n_horizons: int):
    assert n_horizons >= 1, f"n_horizons must be >= 1, got {n_horizons}"
    decoder_params = [
        [
            OrderedDict({'deconv1_leaky_1': [96, 96, 4, 2, 1]}),
            OrderedDict({'deconv2_leaky_1': [96, 96, 4, 2, 1]}),
            OrderedDict({
                'conv3_leaky_1': [64, 16, 3, 1, 1],
                'conv4_final':   [16, n_horizons, 1, 1, 0]
            }),
        ],
        [
            CLSTM_cell(shape=(64,  64),  input_channels=96, filter_size=5, num_features=96),
            CLSTM_cell(shape=(128, 128), input_channels=96, filter_size=5, num_features=96),
            CLSTM_cell(shape=(256, 256), input_channels=96, filter_size=5, num_features=64),
        ]
    ]
    return decoder_params
