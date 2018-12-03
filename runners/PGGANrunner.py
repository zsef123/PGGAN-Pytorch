import os
import time
from glob import glob
from itertools import cycle

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import grad

from utils import export_image
from config import resl_to_batch

class PGGANRunner:
    def __init__(self, arg, G, D, optim_G, optim_D, torch_device, loss, logger):
        self.arg    = arg
        self.device = torch_device
        self.logger = logger
        self.save_dir = arg.save_dir
        
        self.batch = resl_to_batch[arg.start_resl]
        self.G = G
        self.D = D
        self.optim_G = optim_G
        self.optim_D = optim_D

        self.ones  = torch.ones(self.batch)
        self.zeros = torch.zeros(self.batch)

        if loss == "lsgan":
            self.loss = nn.MSELoss()
            self._train_G = self._lsgan_train_G
            self._train_D = self._lsgan_train_D
        elif loss == "wgangp":
            self.loss = None            
            self.gp_lambda = self.arg.gp_lambda
            self._train_G = self._wgangp_train_G
            self._train_D = self._wgangp_train_D
            
        self.best_metric = -1

        self.load() 

    def save(self, epoch, filename):
        """Save current epoch model

        Save Elements:
            model_type : arg.model
            start_epoch : current epoch
            network : network parameters
            optimizer: optimizer parameters
            best_metric : current best score

        Parameters:
            epoch : current epoch
            filename : model save file name
        """
        if epoch < 50:
            return

        torch.save({"model_type_G": self.G,
                    "model_type_D": self.D,
                    "start_epoch" : epoch + 1,
                    "G" : self.G.state_dict(),
                    "D" : self.D.state_dict(),
                    "optim_G" : self.optim_G.state_dict(),
                    "optim_D" : self.optim_D.state_dict(),
                    "best_metric": self.best_metric
                    }, self.save_dir + "/%s.pth.tar"%(filename))
        print("Model saved %d epoch"%(epoch))

    def load(self, filename=None):
        """ Model load. same with save"""
        if filename is None:
            # load last epoch model
            filenames = sorted(glob(self.save_dir + "/*.pth.tar"))
            if len(filenames) == 0:
                print("Not Load")
                return
            else:
                filename = os.path.basename(filenames[-1])

        file_path = self.save_dir + "/" + filename
        if os.path.exists(file_path) is True:
            print("Load %s to %s File"%(self.save_dir, filename))
            ckpoint = torch.load(file_path)
            if ckpoint["model_type_G"] != self.G and ckpoint["model_type_D"] != self.D:
                raise ValueError("Ckpoint Model Type is %s"%(ckpoint["model_type"]))
            
            self.G.load_state_dict(ckpoint["G"])
            self.D.load_state_dict(ckpoint["D"])
            self.optim_G.load_state_dict(ckpoint['optim_G'])
            self.optim_D.load_state_dict(ckpoint['optim_D'])
            self.start_epoch = ckpoint['start_epoch']
            self.best_metric = ckpoint["best_metric"]
            print("Load Model Type : G:%s / D:%s, epoch : %d acc : %f"%(ckpoint["model_type_G"], ckpoint["model_type_D"], self.start_epoch, self.best_metric))
        else:
            print("Load Failed, not exists file")

    def _lsgan_train_G(self, phase):
        latent_init = torch.randn(self.batch, 512, 1, 1).to(self.device)
        fake_x = self.G.forward(latent_init, phase)
        fake_y = self.D.forward(fake_x, phase)

        self.optim_G.zero_grad()
        loss_G = self.loss(fake_y, self.ones)
        loss_G.backward()
        self.optim_G.step()
        return loss_G

    def _lsgan_train_D(self, x, phase):
        latent_init = torch.randn(self.batch, 512, 1, 1).to(self.device)
        fake_x = self.G.forward(latent_init, phase)
        fake_y = self.D.forward(fake_x, phase)
        fake_loss = self.loss(fake_y, self.zeros)

        real_y = self.D.forward(x, phase)
        real_loss = self.loss(real_y, self.ones)

        self.optim_D.zero_grad()
        loss_D = fake_loss + real_loss
        loss_D.backward()
        self.optim_D.step()
        return loss_D


    def _wgangp_train_G(self, phase):
        latent_init = torch.randn(self.batch, 512, 1, 1).to(self.device)
        fake_x = self.G.forward(latent_init, phase)
        fake_y = self.D.forward(fake_x, phase)

        self.optim_G.zero_grad()
        loss_G = -1 * fake_y.mean()
        loss_G.backward()
        self.optim_G.step()
        return loss_G

    def _wgangp_train_D(self, x, phase):
        latent_init = torch.randn(self.batch, 512, 1, 1).to(self.device)
        fake_x = self.G.forward(latent_init, phase)
        fake_y = self.D.forward(fake_x, phase)
        fake_loss = fake_y.mean()

        real_y = self.D.forward(x, phase)
        real_loss = -1 * real_y.mean()
        
        alpha = torch.rand((self.batch, 1, 1, 1)).to(self.device)

        x_hat = alpha * x.cuda() + (1 - alpha) * fake_x
        x_hat.requires_grad_(True)

        pred_hat = self.D.forward(x_hat, "stabilization")
        gradients = grad(outputs=pred_hat, inputs=x_hat,
                         grad_outputs=torch.ones(pred_hat.size()).to(self.device),
                         create_graph=True, retain_graph=True, only_inputs=True)[0]

        gp = self.gp_lambda * ((gradients.view(self.batch, -1).norm(2, 1) - 1) ** 2).mean()

        self.optim_D.zero_grad()
        loss_D = fake_loss + real_loss + gp
        loss_D.backward()
        self.optim_D.step()
        return loss_D


    def train(self, scalable_loader, stab_step, tran_step):
        global_step = 0

        resl = self.arg.start_resl
        loader = scalable_loader(resl)

        print("\n=====================\n", self.G)
        print("\n=====================\n", self.D)

        # Stab for initial resolution (e.g. )
        for step in range(stab_step):
            input_, _ = next(loader)
            loss_D = self._train_D(input_, "stabilization")
            loss_G = self._train_G("stabilization")
            if (step + 1) % 50 == 0 or (global_step + 1) % 50 == 0:
                export_image(self.G, self.save_dir, global_step, resl, step, "stabilization", img_num=10)
                print("Stabilization ... step: %6d / global step : %06d / resl : %d / lossD : %f / lossG : %f" % (step, global_step, resl, loss_D, loss_G))
            global_step += 1

        # Grow
        self.G.grow_network()
        self.D.grow_network()
        print("Grow ... global step: %06d / resl is now %d" % (global_step, resl*2))

        print("\n=====================\n", self.G)
        print("\n=====================\n", self.D)
        
        resl *= 2                               ## make this part as function ?
        loader = scalable_loader(resl)
        self.batch = resl_to_batch[resl]
        self.ones  = torch.ones(self.batch)
        self.zeros = torch.zeros(self.batch)

        while (resl < self.arg.end_resl):
            # Trans
            for _ in range(tran_step):
                input_, _ = next(loader)
                loss_D = self._train_D(input_, "transition")
                loss_G = self._train_G("transition")
                self.G.alpha += 1 / (tran_step)
                self.D.alpha += 1 / (tran_step)
                if (step + 1) % 50 == 0 or (global_step + 1) % 50 == 0:
                    export_image(self.G, self.save_dir, global_step, resl, step, "transition", img_num=10)
                    print("Transition ... step: %6d / global step : %06d / resl : %d / lossD : %f / lossG : %f" % (step, global_step, resl, loss_D, loss_G))
                global_step += 1

            # Stab
            for step in range(stab_step):
                input_, _ = next(loader)
                loss_D = self._train_D(input_, "stabilization")
                loss_G = self._train_G("stabilization")
                if (step + 1) % 50 == 0 or (global_step + 1) % 50 == 0:
                    export_image(self.G, self.save_dir, global_step, resl, step, "stabilization", img_num=10)
                    print("Stabilization ... step: %6d / global step : %06d / resl : %d / lossD : %f / lossG : %f" % (step, global_step, resl, loss_D, loss_G))

                global_step += 1

            # Grow
            self.G.grow_network()
            self.D.grow_network()
            print("Grow ... global step: %06d / resl is now %d" % (global_step, resl*2))
            print("\n=====================\n", self.G)
            print("\n=====================\n", self.D)

            resl *= 2
            loader = scalable_loader(resl)
            self.batch = resl_to_batch[resl]
            self.ones  = torch.ones(self.batch)
            self.zeros = torch.zeros(self.batch)
