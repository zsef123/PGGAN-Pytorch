import torch
import torch.nn as nn
import torch.nn.functional as F


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
        
        # z shape is [G C H W] -> [G 1 C H W]
        if batch == 1:
            x = input_
        else:
            xs = [z.unsqueeze(dim=1) for z in torch.split(input_, self.group)]
            x = torch.cat(xs, dim=1)
            x = torch.std(x,  dim=0, unbiased=False)
        # [C H W]-wise mean
        x = F.adaptive_avg_pool3d(x, 1)
        # M 1 1 1 -> N 1 H W
        x = x.repeat([self.group, 1, *input_.shape[2:]])
        x = torch.cat([input_, x], dim=1)
        return x
