import torch
import torch.nn as nn

class EqualizedLRLayer(nn.Module):
    def __init__(self, layer, scale_mode="fan_in", gain=2):
        super().__init__()

        self.layer = layer
        
        self.bias = self.layer.bias
        self.layer.bias = None
        
        if self.layer.weight.ndimension() > 2:
            expand_ch = [1] * (self.layer.weight.ndimension() - 2)
            self.bias = nn.Parameter(self.bias.view(1, self.bias.shape[0], *expand_ch))

        nn.init.normal_(self.layer.weight)
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(self.layer.weight)
        if scale_mode == "fan_in":
            self.scale = (gain / fan_in) ** 0.5
        elif scale_mode == "fan_out":
            self.scale = (gain / fan_out) ** 0.5

    def forward(self, x):
        x = self.layer(x) * self.scale
        x = x + self.bias
        return x
