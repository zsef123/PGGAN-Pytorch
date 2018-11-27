import torch
import torch.nn as nn
import torch.nn.init

# Wrapping of original layers (e.g. Conv2d, Linear, ...)
# Bias of original layers should be False maybe?
class EqualizedLRLayer(nn.Module):
    def __init__(self, layer, bias=True, mode="fan_in"):

        if layer.bias is not None:
            print("Warning : bias of layer to be wrapped by 'EqualizedLRLayer' must not exist. Forced replace occured")
            layer.bias = None
        assert mode in ["fan_in", "fan_out"], "mode can be 'fan_in' or 'fan_out' either, not %s" % mode
        super().__init__()

        self.layer = layer
        nn.init.normal_(self.layer.weight)

        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(layer.weight)

        if mode == "fan_in":
            self.scale = (2 / fan_in) ** 0.5
        elif mode == "fan_out":
            self.scale = (2 / fan_out) ** 0.5

        if bias:
            self.bias = torch.zeros(layer.weight.size(0))
        else:
            self.bias = 0
    
    def forward(self, x):
        x = (self.layer(x) * self.scale) + self.bias.view(1, self.bias.size(0), 1, 1)
        return x

if __name__ == "__main__":
    eqconv = EqualizedLRLayer(nn.Conv2d(3, 5, 10, bias=False), True)
    input_ = torch.randn(1, 3, 20, 20)
    output_ = eqconv(input_)

    print(output_.size())

