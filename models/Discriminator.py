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

class _DownReslBlock(nn.Module):
    def __init__(self, resl):
        super().__init__()
        self.resl = resl
        # in_c, out_c is define by resl
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

class D(nn.Module):
    def __init__(self, resl=4, group_size=1):
        super().__init__()
        self.resl = resl

        in_c, out_c = resl_to_ch[resl]
        self.resl_blocks = nn.Sequential(nn.Sequential(
            MiniBatchSTD(group_size=group_size),
            EqualizedLRLayer(nn.Conv2d(in_c + 1, out_c, 3, 1, 1, bias=False)),
            nn.LeakyReLU(inplace=True),
            EqualizedLRLayer(nn.Conv2d(in_c, out_c, 4, 1, 1, bias=False)),
            nn.LeakyReLU(inplace=True),
            nn.AdaptiveAvgPool2d(out_c),
            EqualizedLRLayer(nn.Linear(out_c, 1, bias=False))
        ))

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
        self.resl_blocks = nn.Sequential([_DownReslBlock(self.resl), *self.resl_blocks])
        self.rgb_l = self.rgb_h
        self.rgb_h = FromRGBLayer(self.resl)
        self.alpha = 0

    def transition_forward(self, x):
        # low resl path
        x_down = F.avg_pool2d(x, kernel_size=2)
        rgb_l = self.rgb_l(x_down)

        # high resolution path
        rgb_h = self.rgb_h(x)
        x_high = self.resl_blocks[0](rgb_h)

        x = (self.alpha * x_high) + ((1 - self.alpha) * rgb_l)

        for resl in self.resl_blocks[1:]:
            x = resl(x)
        return x

    def stabilization_forward(self, x):
        x = self.rgb_h(x)
        for resl in self.resl_blocks:
            x = resl(x)
        return x
