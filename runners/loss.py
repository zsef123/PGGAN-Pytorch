import torch
import torch.nn as nn
from torch.autograd import grad

class LSGANLoss:
    def __init__(self, batch, device):
        self.mse = nn.MSELoss()
        self.ones  = torch.ones(batch).to(device)
        self.zeros = torch.zeros(batch).to(device)
    
    def loss_D(self, real_y, fake_y, **kwargs):
        fake_loss = self.mse(fake_y, self.zeros)
        real_loss = self.mse(real_y, self.ones)
        return fake_loss + real_loss

    def loss_G(self, fake_y):        
        return self.mse(fake_y, self.ones)


class WGANGPLoss:
    def __init__(self, batch, lambda_, device):
        self.batch = batch
        self.lambda_ = lambda_
        self.device = device

    def loss_D(self, real_y, fake_y, D, real_x, fake_x):
        fake_loss = fake_y.mean()
        real_loss = real_y.mean()
        return fake_loss - real_loss

    def loss_G(self, fake_y):        
        return -fake_y.mean()

    def get_gp(self, D, real_x, fake_x):
        # gradient penalty
        alpha = torch.rand((self.batch, 1, 1, 1)).to(self.device)

        x_hat = alpha * real_x + (1 - alpha) * fake_x
        x_hat.requires_grad_(True)

        pred_hat = D(x_hat)
        gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).to(self.device),
                        create_graph=True, retain_graph=True, only_inputs=True)[0]

        gp = self.lambda_ * ((gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()
        return gp
