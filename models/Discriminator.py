import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Minibatch_stddev import MiniBatchSTD
from models.EqualizedLR import EqualizedLRLayer
from config import resl_to_ch


class FromRGBLayer(nn.Module):
    def __init__(self, resl):
        super().__init__()

        _, out_c = resl_to_ch[resl]
        self.conv = EqualizedLRLayer(nn.Conv2d(3, out_c, 1, bias=False))

    def forward(self, x):
        return self.conv(x)


class DownReslLayer(nn.Module):
    def __init__(self, resl, second_conv_kernel=3):
        super().__init__()

        out_c, in_c  = resl_to_ch[resl]
        self.conv = nn.Sequential(
            EqualizedLRLayer(nn.Conv2d(in_c,  out_c, 3, 1, 1, bias=False)),
            nn.LeakyReLU(inplace=True),
            EqualizedLRLayer(nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool2d(2)
        )
    
    def forward(self, x):
        return self.conv(x)

class DownReslBlock(nn.Module):
    def __init__(self, resl, grop_size):
        super().__init__()

        self.layers = nn.Sequential()
        self.layers.add_module("miniBatchSTD", MiniBatchSTD(group_size=group_size))
        self.layers.add_module("%d_DownReslLayer"%(resl), DownReslLayer(self.resl, second_conv_kernel=4))
        self.first_layer = None

    def grow_network(self, resl):
        key = "%d_DownReslLayer"%(resl)
        self.layers.add_module(key, self.first_layer(self.resl))
        self.layers._modules.move_to_end(key, last=False)
        self.first_layer = DownReslLayer(resl)
    
    def forward(self, x):
        return self.layers(x)


class D(nn.Module):
    def __init__(self, resl=4, group_size=1):
        super().__init__()
        self.resl = resl

        in_c, out_c = resl_to_ch[resl]
        self.resl_blocks = DownReslBlock(resl, group_size)

        self.rgb_l = None
        self.rgb_h = FromRGBLayer(resl)
        self.alpha = 0
               
    def forward(self, x, phase):
        if phase == "transition":
            return self.transition_forward(x)
        elif phase == "stabilization":
            return self.stabilization_forward(x)

    def grow_network(self):
        self.resl *= 2
        self.resl_blocks.grow_network(self.resl)
        self.rgb_l = self.rgb_h
        self.rgb_h = FromRGBLayer(self.resl)
        self.alpha = 0

    def transition_forward(self, x):
        # low resl path
        x_down = F.avg_pool2d(x, kernel_size=2)
        rgb_l = self.rgb_l(x_down)

        # high resolution path
        rgb_h = self.rgb_h(x)
        x_high = self.resl_blocks.first_layer(rgb_h)

        x = (self.alpha * x_high) + ((1 - self.alpha) * rgb_l)
        x = self.resl_blocks(x)        
        return x

    def stabilization_forward(self, x):
        x = self.rgb_h(x)
        x = self.resl_blocks(x)
        return x
