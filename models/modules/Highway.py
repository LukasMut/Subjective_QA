__all__ = ['Highway']

import torch
import torch.nn as nn
import torch.nn.functional as F

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
is_cuda = torch.cuda.is_available()

if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")
"""

class Highway(nn.Module):

    def __init__(
                 self,
                 input_size:int,
    ):
        super(Highway, self).__init__()
        self.input_size = input_size
        self.proj = nn.Linear(self.input_size, self.input_size)
        nn.init.xavier_uniform_(self.proj.weight)
        self.transform = nn.Linear(self.input_size, self.input_size)
        # transform gate bias should be initialized to -2 or -4 according to original paper (i.e., https://arxiv.org/pdf/1505.00387.pdf)
        self.transform.bias.data.fill_(-2.0)

    def forward(self, input):
        """
            y = H(x, W_H) * T(x, W_Y) + x * (1 - T(x, W_T))
            H(x, W_H) = projection layer (affine transform followed by ReLu non-linearity)
            T(x, W_H) = transform gate (affine transform followed by sigmoid non-linearity)
            y = highway_out
        """
        proj_gate = F.relu(self.proj(input))
        transform_gate = torch.sigmoid(self.transform(input))
        # sum of two Hadamard products (elementwise matrix multiplications)
        highway_out = torch.add(torch.mul(proj_gate, transform_gate), torch.mul(input, (1.0 - transform_gate)))
        return highway_out