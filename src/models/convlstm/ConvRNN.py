#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   ConvRNN.py
@Time    :   2020/03/09
@Author  :   jhhuang96
@Mail    :   hjh096@126.com
@Version :   1.0
@Description:   convrnn cell (device-safe version)
'''

#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   ConvRNN.py
@Time    :   2020/03/09
@Author  :   jhhuang96
@Mail    :   hjh096@126.com
@Version :   1.0
@Description:   convrnn cell (device + seq_len safe)
'''

import torch
import torch.nn as nn


class CGRU_cell(nn.Module):
    """
    ConvGRU Cell
    """
    def __init__(self, shape, input_channels, filter_size, num_features):
        super(CGRU_cell, self).__init__()
        self.shape = shape
        self.input_channels = input_channels
        self.filter_size = filter_size
        self.num_features = num_features
        self.padding = (filter_size - 1) // 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                self.input_channels + self.num_features,
                2 * self.num_features,
                self.filter_size,
                1,
                self.padding
            ),
            nn.GroupNorm(2 * self.num_features // 32, 2 * self.num_features)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                self.input_channels + self.num_features,
                self.num_features,
                self.filter_size,
                1,
                self.padding
            ),
            nn.GroupNorm(self.num_features // 32, self.num_features)
        )

    def forward(self, inputs=None, hidden_state=None, seq_len=10):
        # 🔥 FIX 1: if inputs exist, derive seq_len from inputs
        if inputs is not None:
            seq_len = inputs.size(0)

        if hidden_state is None:
            device = inputs.device
            htprev = torch.zeros(
                inputs.size(1),
                self.num_features,
                self.shape[0],
                self.shape[1],
                device=device
            )
        else:
            htprev = hidden_state

        output_inner = []

        for index in range(seq_len):
            if inputs is None:
                x = torch.zeros(
                    htprev.size(0),
                    self.input_channels,
                    self.shape[0],
                    self.shape[1],
                    device=htprev.device
                )
            else:
                x = inputs[index, ...]

            combined_1 = torch.cat((x, htprev), dim=1)
            gates = self.conv1(combined_1)

            zgate, rgate = torch.split(gates, self.num_features, dim=1)
            z = torch.sigmoid(zgate)
            r = torch.sigmoid(rgate)

            combined_2 = torch.cat((x, r * htprev), dim=1)
            ht = self.conv2(combined_2)
            ht = torch.tanh(ht)

            htnext = (1 - z) * htprev + z * ht
            output_inner.append(htnext)
            htprev = htnext

        return torch.stack(output_inner), htnext


class CLSTM_cell(nn.Module):
    """
    ConvLSTM Cell
    """
    def __init__(self, shape, input_channels, filter_size, num_features):
        super(CLSTM_cell, self).__init__()

        self.shape = shape  # (H, W)
        self.input_channels = input_channels
        self.filter_size = filter_size
        self.num_features = num_features
        self.padding = (filter_size - 1) // 2

        self.conv = nn.Sequential(
            nn.Conv2d(
                self.input_channels + self.num_features,
                4 * self.num_features,
                self.filter_size,
                1,
                self.padding
            ),
            nn.GroupNorm(4 * self.num_features // 32, 4 * self.num_features)
        )

    def forward(self, inputs=None, hidden_state=None, seq_len=10):
        # 🔥 FIX 1: if inputs exist, derive seq_len from inputs
        if inputs is not None:
            seq_len = inputs.size(0)

        if hidden_state is None:
            device = inputs.device
            hx = torch.zeros(
                inputs.size(1),
                self.num_features,
                self.shape[0],
                self.shape[1],
                device=device
            )
            cx = torch.zeros(
                inputs.size(1),
                self.num_features,
                self.shape[0],
                self.shape[1],
                device=device
            )
        else:
            hx, cx = hidden_state

        output_inner = []

        for index in range(seq_len):
            if inputs is None:
                x = torch.zeros(
                    hx.size(0),
                    self.input_channels,
                    self.shape[0],
                    self.shape[1],
                    device=hx.device
                )
            else:
                x = inputs[index, ...]

            combined = torch.cat((x, hx), dim=1)
            gates = self.conv(combined)

            ingate, forgetgate, cellgate, outgate = torch.split(
                gates, self.num_features, dim=1
            )

            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)

            output_inner.append(hy)
            hx = hy
            cx = cy

        return torch.stack(output_inner), (hy, cy)

