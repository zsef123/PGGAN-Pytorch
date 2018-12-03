from multiprocessing import Process
import os
import json
import argparse
import torch
import torch.nn as nn
#torch.backends.cudnn.benchmark = True
torch.backends.cudnn.benchmark = False

from datas.ScalableLoader import ScalableLoader

from models.Generator import G
from models.Discriminator import D

from runners.PGGANrunner import PGGANRunner

import utils
from Logger import Logger

def get_optim(net, optim_type, lr, beta, decay, momentum, nesterov):
    optim = {
        "adam" : torch.optim.Adam(net.parameters(), lr=lr, betas=beta, weight_decay=decay),
        "sgd"  : torch.optim.SGD(net.parameters(),
                                 lr=lr, momentum=momentum,
                                 weight_decay=decay, nesterov=True)
    }[optim_type]

    return optim

"""parsing and configuration"""
def arg_parse():
    # projects description
    desc = "PGGAN"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--gpus', type=str, default="0,1,2,3,4,5,6,7",
                        help="Select GPU Numbering | 0,1,2,3,4,5,6,7 | ")
    parser.add_argument('--cpus', type=int, default="16",
                        help="Select CPU Number workers")
    parser.add_argument('--data', type=str, default="",
                        help="data directory name in dataset")

    parser.add_argument('--save_dir', type=str, default='',
                        help='Directory name to save the model')

    parser.add_argument('--epoch', type=int, default=500, help='The number of epochs')
    parser.add_argument('--stab_step', type=int, default=10000, help='The number of stabilization step')
    parser.add_argument('--tran_step', type=int, default=10000, help='The number of transition step')

    parser.add_argument('--optim_G', type=str, default='adam', choices=["adam", "sgd"])
    parser.add_argument('--optim_D', type=str, default='adam', choices=["adam", "sgd"])

    parser.add_argument('--loss', type=str, default='wgangp', choices=["wgangp", "lsgan"])

    parser.add_argument('--start_resl', type=float, default=4)
    parser.add_argument('--end_resl',   type=float, default=1024)

    parser.add_argument('--lr',   type=float, default=0.001)
    # Adam Optimizer
    parser.add_argument('--beta',  nargs="*", type=float, default=(0.5, 0.999))
    # SGD Optimizer
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--decay',    type=float, default=1e-4)
    # Gradient Panelty
    parser.add_argument('--gp_lambda', type=float, default=10.0, help='lambda for Gradient Panelty')
    
    return parser.parse_args()


if __name__ == "__main__":
    arg = arg_parse()

    arg.save_dir = "%s/outs/%s"%(os.getcwd(), arg.save_dir)
    if os.path.exists(arg.save_dir) is False:
        os.mkdir(arg.save_dir)
    
    logger = Logger(arg.save_dir)
    logger.will_write(str(arg) + "\n")

    os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpus
    torch_device = torch.device("cuda")

    # TODO: specify this
    data_path = "../dataset/celeba-1024"
    print("Data Path : ", data_path)
    
    loader = ScalableLoader(data_path, shuffle=True, drop_last=False, num_workers=arg.cpus, shuffled_cycle=True)

    G = G()
    D = D()
    
    G = nn.DataParallel(G).to(torch_device)
    D = nn.DataParallel(D).to(torch_device)
    
    optim_G = get_optim(G, arg.optim_G, arg.lr, arg.beta, arg.decay, arg.momentum, nesterov=True)
    optim_D = get_optim(D, arg.optim_G, arg.lr, arg.beta, arg.decay, arg.momentum, nesterov=True)

    model = PGGANRunner(arg, G, D, optim_G, optim_D, torch_device, arg.loss, logger)
    model.train(loader, arg.stab_step, arg.tran_step)
