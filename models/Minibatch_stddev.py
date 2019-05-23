import torch
import torch.nn as nn


class MiniBatchSTD(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        std = torch.std(x).expand(x.shape[0], 1, *x.shape[2:])
        return torch.cat([x, std], dim=1)
