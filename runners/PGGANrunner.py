import os
import time
from glob import glob
from itertools import cycle

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import grad

from preset import resl_to_batch, resl_to_lr

from runners.train_step import Train_LSGAN, Train_WGAN_GP


def get_optim(net, optim_type, resl, beta, decay, momentum, nesterov=True):
    lr = resl_to_lr[resl]
    optim = {
        "adam" : torch.optim.Adam(net.parameters(), lr=lr, betas=beta, weight_decay=decay),
        "sgd"  : torch.optim.SGD(net.parameters(),
                                lr=lr, momentum=momentum,
                                weight_decay=decay, nesterov=True)
    }[optim_type]

    return optim


class PGGANrunner:
    def __init__(self, arg, G, D, scalable_loader, torch_device, loss, tensorboard):
        self.arg    = arg
        self.device = torch_device
        self.save_dir = arg.save_dir
        self.scalable_loader = scalable_loader
        
        self.img_num   = arg.img_num
        self.batch     = resl_to_batch[arg.start_resl]
        self.tran_step = self.img_num // self.batch
        self.stab_step = self.img_num // self.batch

        self.G = G
        self.D = D
        self.optim_G = get_optim(self.G, self.arg.optim_G, self.arg.start_resl, self.arg.beta, self.arg.decay, self.arg.momentum)
        self.optim_D = get_optim(self.D, self.arg.optim_G, self.arg.start_resl, self.arg.beta, self.arg.decay, self.arg.momentum)

        self.tensorboard = tensorboard

        if loss == "lsgan":
            self.step = Train_LSGAN(self.G, self.D, self.optim_G, self.optim_D,
                                    self.batch, self.device)
        elif loss == "wgangp":
            self.step = Train_WGAN_GP(self.G, self.D, self.optim_G, self.optim_D,
                                     self.arg.gp_lambda, self.batch, self.device)
            
        self.load_resl = -1
        self.load_global_step = -1
        self.load()

    def save(self, global_step, resl, mode):
        """Save current step model

        Save Elements:
            model_type : arg.model
            start_step : current step
            network : network parameters
            optimizer: optimizer parameters
            best_metric : current best score

        Parameters:
            step : current step
            filename : model save file name
        """
        torch.save({"global_step" : global_step,
                    "resl" : resl,
                    "G" : self.G.state_dict(),
                    "D" : self.D.state_dict(),
                    "optim_G" : self.optim_G.state_dict(),
                    "optim_D" : self.optim_D.state_dict(),
                    }, self.save_dir + "/step_%07d_resl_%d_%s.pth.tar"%(global_step, resl, mode))
        print("Model saved %d step"%(global_step))

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
            self.G.load_state_dict(ckpoint["G"])
            self.D.load_state_dict(ckpoint["D"])
            self.optim_G.load_state_dict(ckpoint['optim_G'])
            self.optim_D.load_state_dict(ckpoint['optim_D'])
            self.load_global_step = ckpoint["global_step"]
            self.load_resl = ckpoint["resl"]
            print("Load Model, Global step : %d / Resolution : %d " % (self.load_global_step, self.load_resl))
        else:
            print("Load Failed, not exists file")

    def grow_architecture(self, resl):
        if resl < self.load_resl:
            return resl * 2

        resl *= 2

        self.G.module.grow_network()
        self.D.module.grow_network()

        self.G.to(self.device)
        self.D.to(self.device)

        torch.cuda.empty_cache()

        self.batch     = resl_to_batch[resl]
        self.stab_step = self.img_num // self.batch
        self.tran_step = self.img_num // self.batch
 
        optim_G = get_optim(self.G, self.arg.optim_G, resl, self.arg.beta, self.arg.decay, self.arg.momentum)
        optim_D = get_optim(self.D, self.arg.optim_D, resl, self.arg.beta, self.arg.decay, self.arg.momentum)
        self.step.grow(self.batch, optim_G, optim_D)
        
        return resl

    def train(self):
        # Initialize Train
        global_step, resl = 0, self.arg.start_resl
        loader = self.scalable_loader(resl)

        def _step(step, input_, mode, LOG_PER_STEP=10):            
            nonlocal global_step
            if global_step <= self.load_global_step:
                global_step += 1
                return

            input_ = input_.to(self.device)
            log_D = self.step.train_D(input_, mode)
            log_G = self.step.train_G(mode)
            
            # Save images and record logs
            if (step % LOG_PER_STEP) == 0:
                print("[% 6d/% 6d : % 3.2f %%]" % (step, self.tran_step, (step / self.tran_step) * 100))
                self.tensorboard.log_image(self.G, mode, resl, global_step)
                self.tensorboard.log_scalar("Loss/%d"%(resl), {**log_D, **log_G}, global_step)
                self.save(global_step, resl, mode)
                # self.tensorboard.log_hist(self.G.module, global_step)
                # self.tensorboard.log_hist(self.D.module, global_step)
            global_step += 1

        # Stabilization on initial resolution (default: 4 * 4)
        for step in range(self.stab_step):
            input_, _ = next(loader)
            _step(step, input_ , "stabilization")

        while (resl < self.arg.end_resl):
            # Grow and update resolution, batch size, etc. Load the models on GPUs
            resl = self.grow_architecture(resl)
            loader = self.scalable_loader(resl)
            
            # Transition
            for step in range(self.tran_step):
                input_, _ = next(loader)
                _step(step, input_, "transition")
                self.G.module.update_alpha(1 / self.tran_step)
                self.D.module.update_alpha(1 / self.tran_step)

            # Stabilization
            for step in range(self.stab_step):
                input_, _ = next(loader)
                _step(step, input_, "stabilization")
            
