import torch
import torch.nn as nn
import torch.nn.functional as F
from models.PixelWiseNorm import PixelWiseNormLayer
from config import resl_to_ch

class ToRGBLayer(nn.Module):
    def __init__(self, resl):
        super().__init__()
        _, in_c  = resl_to_ch[resl]

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, 3, 1),
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
            nn.Conv2d(in_c, out_c,  3, 1, 1),
            PixelWiseNormLayer(),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, 1, 1),
            PixelWiseNormLayer(),
            nn.LeakyReLU(inplace=True),
        )
    
    def forward(self, x):
        up   = F.interpolate(x, scale_factor=2)
        conv = self.conv(up)
        return conv

class G:
    def __init__(self, resl=4):
        self.resl = resl

        in_c, out_c = resl_to_ch[resl]
        self.resl_blocks = [nn.Sequential(
            nn.ConvTranspose2d(in_c,  out_c, 4),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, 1, 1),
            PixelWiseNormLayer(),
            nn.LeakyReLU(inplace=True),
        )]

        self.rgb_l = None
        self.rgb_h = ToRGBLayer(self.resl)

        self.alpha = 0
        
    def grow_network(self):
        self.resl *= 2
        self.resl_blocks.append(ReslBlock(self.resl))        
        self.rgb_l = self.rgb_h
        self.rgb_h = ToRGBLayer(self.resl)
        self.alpha = 0

    def transition_train(self, x):        
        for resl in self.resl_blocks[:-1]:      # 마지막 것만 빼고 forward
            x = resl(x)

        # low resolution path
        x_up = F.interpolate(x, scale_factor=2)        
        rgb_l = self.rgb_l(x_up)               # 이전 해상도의 rgb block 에 forward

        # high resolution path
        x = self.resl_blocks[-1](x)
        rgb_h = self.rgb_h(x)                   # 현재 해상도의 rgb block 에 forward
        return (self.alpha * rgb_h) + ((1 - self.alpha) * rgb_l)

    def stabilize_train(self, x):
        for resl in self.resl_blocks:
            x = resl(x)
        rgb_h = self.rgb_h(x)
        return rgb_h

    