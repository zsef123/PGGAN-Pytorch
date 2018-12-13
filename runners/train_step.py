import torch
import torch.nn as nn
from torch.autograd import grad

class Train_LSGAN:
    def __init__(self, G, D, optim_G, optim_D, batch, device):
        self.G = G
        self.D = D
        self.optim_G = optim_G
        self.optim_D = optim_D

        self.loss = nn.MSELoss()
        self.device = device

        self.batch = batch
        self.ones  = torch.ones(batch).to(device)
        self.zeros = torch.zeros(batch).to(device)
    
    def train_D(self, x, mode):
        latent_init = torch.randn(self.batch, 512, 1, 1).to(self.device)

        fake_x = self.G.forward(latent_init, mode)
        fake_y = self.D.forward(fake_x, mode)
        fake_y = fake_y.squeeze()
        fake_loss = self.loss(fake_y, self.zeros)

        real_y = self.D.forward(x, mode)
        real_y = real_y.squeeze()
        real_loss = self.loss(real_y, self.ones)

        self.optim_D.zero_grad()
        loss_D = fake_loss + real_loss
        loss_D.backward()
        self.optim_D.step()
        return loss_D

    def train_G(self, mode):
        latent_init = torch.randn(self.batch, 512, 1, 1).to(self.device)

        fake_x = self.G.forward(latent_init, mode)
        fake_y = self.D.forward(fake_x, mode)
        fake_y = fake_y.squeeze()

        self.optim_G.zero_grad()
        loss_G = self.loss(fake_y, self.ones)
        loss_G.backward()
        self.optim_G.step()
        return loss_G

    def grow(self, batch, optim_G, optim_D):
        self.batch   = batch
        self.optim_G = optim_G
        self.optim_D = optim_D
        self.ones  = torch.ones(batch).to(self.device)
        self.zeros = torch.zeros(batch).to(self.device)


class Train_WGAN_GP:
    def __init__(self, G, D, optim_G, optim_D, gp_lambda, batch, device):
        self.G = G
        self.D = D
        self.optim_G = optim_G
        self.optim_D = optim_D

        self.device = device

        self.batch = batch
        self.gp_lambda = gp_lambda

    def get_gp(self, x, fake_x):
        alpha = torch.rand((self.batch, 1, 1, 1)).to(self.device)
        x_hat = alpha * x + (1 - alpha) * fake_x
        
        x_hat.requires_grad_(True)

        pred_hat = self.D.forward(x_hat, "stabilization")
        gradients = grad(outputs=pred_hat, inputs=x_hat,
                         grad_outputs=torch.ones(pred_hat.size()).to(self.device),
                         create_graph=True, retain_graph=True, only_inputs=True)[0]
        gpgp_yaya = (gradients.view(self.batch, -1).pow(2).sum(dim=1) + 1e-4).sqrt() - 1
        # gp = self.gp_lambda * ((gradients.view(self.batch, -1).norm(2, 1) - 1) ** 2).mean()
        gp = self.gp_lambda * (gpgp_yaya ** 2).mean()

        return gp

    def train_D(self, x, mode):
        latent_init = torch.randn(self.batch, 512, 1, 1).to(self.device)

        fake_x = self.G.forward(latent_init, mode)
        fake_y = self.D.forward(fake_x, mode)
        fake_loss = fake_y.mean()

        real_y = self.D.forward(x, mode)
        real_loss = -1 * real_y.mean()
        
        gp = self.get_gp(x, fake_x)
    
        self.optim_D.zero_grad()
        loss_D = fake_loss + real_loss + gp
        loss_D.backward()
        self.optim_D.step()
        return loss_D

    def train_G(self, mode):        
        latent_init = torch.randn(self.batch, 512, 1, 1).to(self.device)
        fake_x = self.G.forward(latent_init, mode)
        fake_y = self.D.forward(fake_x, mode)

        self.optim_G.zero_grad()
        loss_G = -1 * fake_y.mean()
        loss_G.backward()
        self.optim_G.step()
        return loss_G
    
    def grow(self, batch, optim_G, optim_D):
        self.batch   = batch
        self.optim_G = optim_G
        self.optim_D = optim_D