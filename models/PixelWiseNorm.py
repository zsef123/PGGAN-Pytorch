import torch
import torch.nn as nn


class PixelWiseNormLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-38

    def forward(self, x):
        x = x.type(torch.float64)
        z = torch.mean(x ** 2, dim=1, keepdim=True)
        x = x * (torch.rsqrt(z + self.eps))
        x = x.type(torch.float32)
        return x
        