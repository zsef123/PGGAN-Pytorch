import os
import numpy as np
from glob import glob
import torch
import torch.nn as nn
from .BaseRunner import BaseRunner
from sklearn.metrics import confusion_matrix
from collections import defaultdict
import time
from utils import get_confusion
from itertools import cycle

class BacRunner:
    def __init__(self, arg, G, D, optim_G, optim_D, torch_device, loss, logger):
        self.arg    = arg
        self.device = torch_device
        self.logger = logger
        
        self.G = G
        self.D = D
        self.resl = arg.resl
        self.loss = loss
        self.optim_G = optim_G
        self.optim_D = optim_D
        self.ones  = torch.ones(self.arg.batch)
        self.zeros = torch.zeros(self.arg.batch)

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

        torch.save({"model_type" : self.model_type,
                    "start_epoch" : epoch + 1,
                    "G" : self.G.state_dict(),
                    "D" : self.D.state_dict(),
                    "optimizer" : self.optim.state_dict(),
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
            if ckpoint["model_type"] != self.model_type:
                raise ValueError("Ckpoint Model Type is %s"%(ckpoint["model_type"]))
            
            self.G.load_state_dict(ckpoint["G"])
            self.D.load_state_dict(ckpoint["D"])
            self.optim.load_state_dict(ckpoint['optimizer'])
            self.start_epoch = ckpoint['start_epoch']
            self.best_metric = ckpoint["best_metric"]
            print("Load Model Type : %s, epoch : %d acc : %f"%(ckpoint["model_type"], self.start_epoch, self.best_metric))
        else:
            print("Load Failed, not exists file")

    def _train_G(self, mode):
        latent_init = torch.randn(self.arg.batch, 512, 1, 1).to(self.device)
        fake_x = self.G.forward(latent_init, mode)
        fake_y = self.D.forward(fake_x, mode)

        self.optim_G.zero_grad()
        loss_G = torch.sum(self.loss(fake_y, self.ones))
        loss_G.backward()
        self.optim_G.step()
        return loss_G

    def _train_D(self, x, mode):
        latent_init = torch.randn(self.arg.batch, 512, 1, 1).to(self.device)
        fake_x = self.G.forward(latent_init, mode)
        fake_y = self.D.forward(fake_x, mode)
        fake_loss = torch.sum(self.loss(fake_y, self.zeros))

        real_y = self.D.forward(x, mode)
        real_loss = torch.sum(self.loss(real_y, self.ones))

        self.optim_D.zero_grad()
        loss_D = fake_loss + real_loss
        loss_D.backward()
        self.optim_D.step()
        return loss_D

    def train(self, scalable_loader, stab_step, tran_step):
        global_step = 0

        resl = self.arg.start_resl
        loader = scalable_loader(resl)

        # Stab for initial resolution (e.g. )
        for _ in range(stab_step):
            input_, _ = next(loader)
            self._train_D(input_, "stabilization")
            self._train_G("stabilization")

        # Grow
        self.G.grow_network()
        self.D.grow_network()

        resl *= 2; loader = scalable_loader(resl)
        while (resl < self.arg.end_resl):
            # Trans
            for _ in range(tran_step):
                # TODO : is cyclic??
                input_, _ = next(loader)
                self._train_D(input_, "transition")
                self._train_G("transition")
                self.G.alpha += 1 / (tran_step)
                self.D.alpha += 1 / (tran_step)

            # Stab
            for _ in range(stab_step):
                input_, _ = next(loader)
                self._train_D(input_, "stabilization")
                self._train_G("stabilization")

            # Grow
            self.G.grow_network()
            self.D.grow_network()
