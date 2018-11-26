import time
import torch
from models.Generator import G
from models.Discriminator import D


if __name__ == "__main__":

    dims = [4, 8, 16, 32, 64, 128, 256, 512]
    imgs = [torch.randn(1, 3, z, z) for z in dims]
    stabilization_step = 1
    transition_step    = 1

    g = G(4)
    d = D(4)
    for full_step, resl in enumerate(dims):
        # stabilization
        for step in range(stabilization_step):
            latent_init = torch.randn(1, 512, 1, 1)
            img = g.stabilize_train(latent_init)
            out = d.stabilize_train(img)

        # grow    
        g.grow_network()
        d.grow_network()
        
        # TODO : 마지막 step 에서는 transition skip
        # transition
        if full_step == len(dims) - 1:
            continue

        for step in range(transition_step):
            latent_init = torch.randn(1, 512, 1, 1)
            img = g.transition_train(latent_init)
            out = d.transition_train(img)
            g.alpha += 1 / (transition_step)
            d.alpha += 1 / (transition_step)
