import torch.nn as nn


class EqualizedLR(nn.Module):
    def __init__(self, layer):
        super().__init__()

        nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

        layer.bias.data.fill_(0)

        self.wscale = layer.weight.data.detach().pow(2.).mean().sqrt()
        layer.weight.data /= self.wscale

        self.layer = layer

    def forward(self, x):
        return self.layer(x * self.wscale)
