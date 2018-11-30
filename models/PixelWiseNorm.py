import torch
import torch.nn as nn


class PixelWiseNormLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-8

    def forward(self, x):         
        z = torch.mean(x.pow(2), dim=1, keepdim=True)
        x = x / (torch.sqrt(z) + self.eps)
        return x