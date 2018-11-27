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
        )
        self.resl = resl

    def forward(self, x):
        return self.conv(x)


class ReslBlock(nn.Module):

    def __init__(self, resl):
        super().__init__()
        self.resl = resl
        in_c, out_c  = resl_to_ch[resl]        
        # print("UpResl resl : ", resl, "in_c : ", in_c, "out_c :", out_c)

        self.conv = nn.Sequential(
            EqualizedLRLayer(nn.Conv2d(in_c, out_c,  3, 1, 1, bias=False)),
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

class G(nn.Module):
    def __init__(self, resl=4):
        super().__init__()
        self.resl = resl

        in_c, out_c = resl_to_ch[resl]
        self.resl_blocks = [nn.Sequential(
            EqualizedLRLayer(nn.ConvTranspose2d(in_c, out_c, 4, bias=False)),
            PixelWiseNormLayer(),
            nn.LeakyReLU(inplace=True),
            EqualizedLRLayer(nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)),
            PixelWiseNormLayer(),
            nn.LeakyReLU(inplace=True),
        )]

        self.rgb_l = None
        self.rgb_h = ToRGBLayer(self.resl)

        self.alpha = 0
        
    def forward(self, x, mode):
        if mode == "transition":
            return self.transition_forward(x)
        elif mode == "stabilization":
            return self.stabilization_forward(x)

    def grow_network(self):
        self.resl *= 2
        self.resl_blocks.append(ReslBlock(self.resl))        
        self.rgb_l = self.rgb_h
        self.rgb_h = ToRGBLayer(self.resl)
        self.alpha = 0

    def transition_forward(self, x):        
        for resl in self.resl_blocks[:-1]:
            x = resl(x)

        # low resolution path
        x_up = F.interpolate(x, scale_factor=2)        
        rgb_l = self.rgb_l(x_up)

        # high resolution path
        x = self.resl_blocks[-1](x)
        rgb_h = self.rgb_h(x)
        return (self.alpha * rgb_h) + ((1 - self.alpha) * rgb_l)

    def stabilization_forward(self, x):
        for resl in self.resl_blocks:
            x = resl(x)
        rgb_h = self.rgb_h(x)
        return rgb_h

    