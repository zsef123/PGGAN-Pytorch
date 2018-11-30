import torch
import torchvision

def export_image(generator, save_path, global_step, resl, step, mode, img_num=10):
    latent_z = torch.randn(img_num, 512, 1, 1).to(self.device)
    generated_img = generator.stabilization_forward()

    for idx, img in enumerate(generated_img):
        torchvision.utils.save_image(img, "%s/fig/[%03d]_%s_%04d_%02d.png" % (save_path, resl, mode, global_step, idx))     # ~~~/out/fig/{epoch}_{idx}.png
