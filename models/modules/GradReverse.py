__all__ = [
           'GRL',
           'grad_reverse',
]

import torch
import torch.nn as nn

class GRL(torch.autograd.Function):
    lambd = 1.0
    @staticmethod
    def forward(ctx, x):
        # identity function
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # reverse the gradients and scale by the constant lambda
        return GRL.lambd * grad_output.neg()
    
def grad_reverse(x, lambd=1.0):
    GRL.lambd = lambd
    return GRL.apply(x)