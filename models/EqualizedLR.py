import torch
import torch.nn as nn

class EqualizedLRLayer(nn.Module):
    def __init__(self, layer, bias=True, scale_mode="fan_in"):

        if layer.bias is not None:
            print("Warning : bias of layer to be wrapped by 'EqualizedLRLayer' must not exist. Forced replace occured")
            layer.bias = None
        assert scale_mode in ["fan_in", "fan_out"], "scale_mode can be 'fan_in' or 'fan_out' either, not %s" % scale_mode
        super().__init__()

        self.layer = layer
        nn.init.normal_(self.layer.weight)

        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(self.layer.weight)

        if scale_mode == "fan_in":
            self.scale = (2 / fan_in) ** 0.5
        elif scale_mode == "fan_out":
            self.scale = (2 / fan_out) ** 0.5

        if bias:
            bias_data = torch.zeros(layer.weight.size(0))
            self.bias = nn.Parameter(bias_data)
        else:
            self.bias = 0
    
    def forward(self, x):
        x = (self.layer(x) * self.scale) + self.bias
        return x

        
if __name__ == "__main__":
    eqconv = EqualizedLRLayer(nn.Conv2d(3, 5, 10, bias=False), True)
    input_ = torch.randn(1, 3, 20, 20)
    output_ = eqconv(input_)

    print(output_.size())

