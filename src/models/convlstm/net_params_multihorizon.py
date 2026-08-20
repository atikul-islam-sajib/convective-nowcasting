# src/models/convlstm/net_params_multihorizon.py
from collections import OrderedDict
from .ConvRNN import CLSTM_cell

# ============================================================
# ENCODER — fixed for 256x256 input, 3 channels per timestep
# ============================================================
convlstm_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [3, 16, 3, 1, 1]}),
        OrderedDict({'conv2_leaky_1': [64, 64, 3, 2, 1]}),
        OrderedDict({'conv3_leaky_1': [96, 96, 3, 2, 1]}),
    ],
    [
        CLSTM_cell(shape=(256, 256), input_channels=16, filter_size=5, num_features=64),
        CLSTM_cell(shape=(128, 128), input_channels=64, filter_size=5, num_features=96),
        CLSTM_cell(shape=(64,  64),  input_channels=96, filter_size=5, num_features=96),
    ]
]

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
