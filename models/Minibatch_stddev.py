import torch
import torch.nn as nn
import torch.nn.functional as F


# Acknowledgement : referenced https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py
class MiniBatchSTD(nn.Module):
    def __init__(self, group_size=4):
        super().__init__()
        self.group = group_size
        
    def forward(self, input_):
        batch = input_.shape[0]
        if batch % self.group != 0 and batch >= self.group:
            raise ValueError("Batch Size(%d) must be divisible by (or smaller than) Group Size(%d)"%(batch, self.group))
        elif batch < self.group:
            self.group = batch

        if batch == 1:
            x = input_
        else:
            # [NCHW]  Input shape.
            x = input_.reshape(self.group, -1, *input_.shape[1:])   # [GMCHW] Split minibatch into M groups of size G.
            x = x.type(torch.float64)
            x = x - x.mean(dim=0, keepdim=True)                     # [GMCHW] Subtract mean over group.
            x = (x ** 2).mean(dim=0)                                # [MCHW]  Calc variance over group.
            x = torch.sqrt(x + 1e-38)                               # [MCHW]  Calc stddev over group.
        x = torch.mean(x, dim=[1, 2, 3], keepdim=True)              # [M111]  Take average over fmaps and pixels.
        x = x.type_as(input_)
        x = x.repeat([self.group, 1, *input_.shape[2:]])            # [N1HW]  Replicate over group and pixels.
        x = torch.cat([input_, x], dim=1)                           # [NCHW]  Append as new fmap.
        return x
