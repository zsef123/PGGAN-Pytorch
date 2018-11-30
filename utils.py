import os
import torch
import torchvision

def export_image(G, save_path, global_step, resl, step, phase, img_num=10):
    latent_z = torch.randn(img_num, 512, 1, 1).cuda()
    generated_img = G.forward(latent_z, "stabilization")

    for idx, img in enumerate(generated_img):
        if not os.path.exists(save_path + "/fig"):
            os.mkdir(save_path + "/fig")
        torchvision.utils.save_image(img, "%s/fig/[%03d]_%s_%04d_%02d.png" % (save_path, resl, phase, global_step, idx))     # ~~~/out/fig/{epoch}_{idx}.png
