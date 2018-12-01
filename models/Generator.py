import torch
import torch.nn as nn
import torch.nn.functional as F
from models.PixelWiseNorm import PixelWiseNormLayer
from models.EqualizedLR import EqualizedLRLayer
from config import resl_to_ch


class ToRGBLayer(nn.Module):
    def __init__(self, resl):
        super().__init__()

        _, in_c  = resl_to_ch[resl]
        self.conv = nn.Sequential(
            EqualizedLRLayer(nn.Conv2d(in_c, 3, 1, bias=False)),
            nn.Tanh()
        )

    def forward(self, x):
        return self.conv(x)


class ReslLayer(nn.Module):
    def __init__(self, resl, first_conv_kernel=3):
        super().__init__()

        in_c, out_c  = resl_to_ch[resl]        
        self.conv = nn.Sequential(
            EqualizedLRLayer(nn.Conv2d(in_c, out_c, first_conv_kernel, 1, 1, bias=False)),
            PixelWiseNormLayer(),
            nn.LeakyReLU(inplace=True),
            EqualizedLRLayer(nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)),
            PixelWiseNormLayer(),
            nn.LeakyReLU(inplace=True),
        )
    
    def forward(self, x):
        up   = F.interpolate(x, scale_factor=2)
        conv = self.conv(up)
        return conv

class ReslBlock(nn.Module):
    def __init__(self, resl):
        super().__init__()
        
        self.layers = nn.Sequential()
        self.layers.add_module("%d_ReslLayer"%(resl), ReslLayer(self.resl, first_conv_kernel=4))
        self.last_layer = None

    def grow_network(self, resl):
        self.layers.add_module("%d_ReslLayer"%(resl), self.last_layer)
        self.last_layer = ReslLayer(resl)
    
    def transition_forward(self, x):
        x_low = self.layers(x)
        x_high = self.last_layer(x_low)
        return x_low, x_high
    
    def stabilization_forward(self, x):
        x = self.layers(x)
        return x


class G(nn.Module):
    def __init__(self, resl=4, device=None):
        super().__init__()
        self.resl = resl

        in_c, out_c = resl_to_ch[resl]
        self.resl_block = ReslBlock(resl)

        self.rgb_l = None
        self.rgb_h = ToRGBLayer(self.resl)

        self.alpha = 0

        self.device = device
        
    def forward(self, x, phase):
        if phase == "transition":
            return self.transition_forward(x)
        elif phase == "stabilization":
            return self.stabilization_forward(x)

    def grow_network(self):
        self.resl *= 2
        self.resl_block.grow_network(self.resl)
        self.rgb_l = self.rgb_h
        self.rgb_h = ToRGBLayer(self.resl)
        self.alpha = 0

    def transition_forward(self, x):
        x_low, x_high = self.resl_block.transition_forward(x)

        # low resolution path
        x_up = F.interpolate(x_low, scale_factor=2)        
        rgb_l = self.rgb_l(x_up)

        # high resolution path
        rgb_h = self.rgb_h(x_high)
        return (self.alpha * rgb_h) + ((1 - self.alpha) * rgb_l)

    def stabilization_forward(self, x):
        x = self.resl_block(x)
        rgb_h = self.rgb_h(x)
        return rgb_h

    