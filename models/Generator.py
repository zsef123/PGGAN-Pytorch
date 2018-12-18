import torch
import torch.nn as nn
import torch.nn.functional as F

from models.PixelWiseNorm import PixelWiseNormLayer
from models.EqualizedLR import EqualizedLRLayer

from preset import resl_to_ch


class ToRGBLayer(nn.Module):
    def __init__(self, resl):
        super().__init__()
        _, in_c  = resl_to_ch[resl]

        self.conv = nn.Sequential(
            EqualizedLRLayer(nn.Conv2d(in_c, 3, 1)),
            nn.Tanh()
        )

    def forward(self, x):
        return self.conv(x)


class ReslBlock(nn.Module):
    def __init__(self, resl):
        super().__init__()
        in_c, out_c  = resl_to_ch[resl]

        self.conv = nn.Sequential(
            EqualizedLRLayer(nn.Conv2d(in_c, out_c,  3, 1, 1)),
            PixelWiseNormLayer(),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            EqualizedLRLayer(nn.Conv2d(out_c, out_c, 3, 1, 1)),
            PixelWiseNormLayer(),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
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
        self.resl_blocks = nn.Sequential(
            # Original Repo using "tf.nn.conv2d_transpose"
            # Same with EqualizedLRLayer(nn.ConvTranspose2d(in_c, out_c, 4)),
            nn.Upsample(size=(4, 4)),
            EqualizedLRLayer(nn.Conv2d(in_c, out_c, 1, 1, 0)),
            PixelWiseNormLayer(),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            EqualizedLRLayer(nn.Conv2d(out_c, out_c, 3, 1, 1)),
            PixelWiseNormLayer(),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.rgb_l = None
        self.rgb_h = ToRGBLayer(self.resl)
        self.alpha = 0
        
    def forward(self, x, phase):
        if phase == "transition":
            return self.transition_forward(x)
        elif phase == "stabilization":
            return self.stabilization_forward(x)

    def grow_network(self):
        self.resl *= 2
        self.resl_blocks = nn.Sequential(*self.resl_blocks, ReslBlock(self.resl))
        self.rgb_l = self.rgb_h
        self.rgb_h = ToRGBLayer(self.resl)
        self.alpha = 0

    def transition_forward(self, x):
        x = self.resl_blocks[:-1](x)
        
        # experiment candidate : apply rgb_l first and succeeding interpolate
        # low resolution path
        x_up = F.interpolate(x, scale_factor=2)
        rgb_l = self.rgb_l(x_up)

        # high resolution path
        x = self.resl_blocks[-1](x)
        rgb_h = self.rgb_h(x)

        return (self.alpha * rgb_h) + ((1 - self.alpha) * rgb_l)

    def stabilization_forward(self, x):
        x = self.resl_blocks(x)
        rgb_h = self.rgb_h(x)
        return rgb_h

    def update_alpha(self, delta):
        self.alpha += delta
        self.alpha = min(1, self.alpha)
