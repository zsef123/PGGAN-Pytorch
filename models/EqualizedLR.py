import torch
import torch.nn as nn

class EqualizedLRLayer(nn.Module):
    def __init__(self, layer, scale_mode="fan_in"):
        super().__init__()

        self.layer = layer
        if layer.bias is not None:
            layer.bias = None
            channel = layer.weight.shape[0]
            ndim = layer.weight.ndimension()
            bias_data  = torch.zeros(channel)
            if ndim > 2:
                expand_ch = [1] * (ndim - 2)
                bias_data = bias_data.view(channel, *expand_ch)
            self.bias  = nn.Parameter(bias_data)
        else:
            self.bias = 0

        nn.init.normal_(self.layer.weight)

        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(self.layer.weight)
        if scale_mode == "fan_in":
            self.scale = (2 / fan_in) ** 0.5
        elif scale_mode == "fan_out":
            self.scale = (2 / fan_out) ** 0.5

    def forward(self, x):
        # print("======================EQ======================")
        # print("layer            : ", self.layer)
        # print("x.shape          : ", x.shape)
        # print("layer(x) * scale : ", (self.layer(x) * self.scale).shape)
        # print("bias shape", self.bias.shape)
        # print("======================EQ======================")
        x = (self.layer(x) * self.scale) + self.bias
        
        return x
