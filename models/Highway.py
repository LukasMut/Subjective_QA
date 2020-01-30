__all__ = ['HighwayBlock']

import torch
import torch.nn as nn
import torch.nn.functional as F

class HighwayBlock(nn.Module):

    def __init__(self, input_size:int):
        super(Highway, self).__init__()
        self.proj = nn.Linear(input_size, input_size)
        self.transform = nn.Linear(input_size, input_size)
        self.transform.bias.data.fill_(-2.0)

    def forward(self, input):
        """
            y = H(x, W_H) * T(x, W_Y) + x * (1 - T(x, W_T))
            H(x, W_H) = projection layer (affine transform followed by ReLu non-linearity)
            T(x, W_H) = transform gate (affine transform followed by sigmoid non-linearity)
            y = highway_out
        """
        proj_gate = nn.functional.relu(self.proj(input))
        transform_gate = nn.functional.sigmoid(self.transform(input))
        highway_out = (proj_gate * transform_gate) + (input * (1 - transform_gate))
        return highway_out