import torch
import torch.nn as nn


class PixelWiseNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x_square_mean = x.pow(2).mean(dim=1, keepdim=True)
        denom = torch.rsqrt(x_square_mean + 1e-8)
        return x * denom
