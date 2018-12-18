import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Minibatch_stddev import MiniBatchSTD
from models.EqualizedLR import EqualizedLRLayer

from preset import resl_to_ch


class FromRGBLayer(nn.Module):
    def __init__(self, resl):
        super().__init__()
        _, out_c = resl_to_ch[resl]
        self.conv = EqualizedLRLayer(nn.Conv2d(3, out_c, 1))

    def forward(self, x):
        return self.conv(x)


class _DownReslBlock(nn.Module):
    def __init__(self, resl):
        super().__init__()
        out_c, in_c  = resl_to_ch[resl]
        
        self.conv = nn.Sequential(
            EqualizedLRLayer(nn.Conv2d(in_c,  out_c, 3, 1, 1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            EqualizedLRLayer(nn.Conv2d(out_c, out_c, 3, 1, 1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.AvgPool2d(2)
        )

    def forward(self, x):        
        return self.conv(x)


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x.view(x.size(0), -1)


class D(nn.Module):
    def __init__(self, resl=4, group_size=4):
        super().__init__()
        self.resl = resl

        in_c, out_c = resl_to_ch[resl]
        self.resl_blocks = nn.Sequential(
            MiniBatchSTD(group_size=group_size),
            EqualizedLRLayer(nn.Conv2d(in_c + 1, out_c, 3, 1, 1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            EqualizedLRLayer(nn.Conv2d(in_c, out_c, 4, 1, 0)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            Flatten(),
            EqualizedLRLayer(nn.Linear(out_c, 1)),
        )

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
        self.resl_blocks = nn.Sequential(_DownReslBlock(self.resl), *self.resl_blocks)
        self.rgb_l = self.rgb_h
        self.rgb_h = FromRGBLayer(self.resl)
        self.alpha = 0

    def transition_forward(self, x):
        # low resolution path
        x_down = F.avg_pool2d(x, kernel_size=2)
        rgb_l = self.rgb_l(x_down)

        # high resolution path
        rgb_h = self.rgb_h(x)
        x_high = self.resl_blocks[0](rgb_h)

        x = (self.alpha * x_high) + ((1 - self.alpha) * rgb_l)
        x = self.resl_blocks[1:](x)
        return x.view(-1)

    def stabilization_forward(self, x):
        x = self.rgb_h(x)
        x = self.resl_blocks(x)
        return x.view(-1)

    def update_alpha(self, delta):
        self.alpha += delta
        self.alpha = min(1, self.alpha)
