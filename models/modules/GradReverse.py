__all__ = [
           'GRL',
           'grad_reverse',
]

import torch
import torch.nn as nn

#TODO: figure out which of the two versions should be used (do we need scale for our purposes)?
class GRL(Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

def grad_reverse(x):
    return GRL.apply(x)

class GRL(torch.autograd.Function):
    scale = 1.0
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return GRL.scale * grad_output.neg()
    
def grad_reverse(x, scale=1.0):
    GRL.scale = scale
    return GRL.apply(x)