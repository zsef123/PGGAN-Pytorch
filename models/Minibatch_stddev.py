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
        if batch == 1:
            x = input_
        else:
            # Minibatch must be divisible by (or smaller than) group_size.
            # [NCHW]  Input shape.
            # [GMCHW] Split minibatch into M groups of size G.
            # x = torch.tensor([z.unsqueeze(dim=1) for z in torch.split(input_, self.group)])
            x = input_.reshape(self.group, -1, *input_.shape[1:])
            # [GMCHW] Subtract mean over group.
            x = x - x.mean(dim=0, keepdim=True)
            # [MCHW]  Calc variance over group.
            x = x.pow(2).mean(dim=0)
            # [MCHW]  Calc stddev over group.
            x = torch.sqrt(x + 1e-4)
            # [M111]  Take average over fmaps and pixels.
        x = torch.mean(x, dim=[1, 2, 3], keepdim=True)
        # [N1HW]  Replicate over group and pixels.

        # M -> M1HW
        x = x.repeat([self.group, 1, *input_.shape[2:]])
        # [NCHW]  Append as new fmap.
        x = torch.cat([input_, x], dim=1)
        # x = torch.std(x,  dim=0, unbiased=False)
        # x = torch.sum(x, dim=0)
        # print("x.sum", x.shape)
        return x

if __name__ == "__main__":
    x = torch.randn(8, 3, 4, 4)
    m = MiniBatchSTD()
    print(m(x).shape)