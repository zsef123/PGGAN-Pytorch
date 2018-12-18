import os
import json
import argparse

import torch
import torch.nn as nn

from datas.ScalableLoader import ScalableLoader

from models.Generator import G
from models.Discriminator import D
from runners.PGGANrunner import PGGANrunner

import utils

"""parsing and configuration"""
def arg_parse():
    # projects description
    desc = "PGGAN"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--gpus', type=str, default="0,1,2,3,4,5,6,7",
                        help="Select GPU Numbering | 0,1,2,3,4,5,6,7 | ")
    parser.add_argument('--cpus', type=int, default="40",
                        help="Select CPU Number workers")

    parser.add_argument('--save_dir', type=str, default='',
                        help='Directory name to save the model')

    parser.add_argument('--img_num', type=int, default=800000, help='The number of images to be used for each phase')

    parser.add_argument('--optim_G', type=str, default='adam', choices=["adam", "sgd"])
    parser.add_argument('--optim_D', type=str, default='adam', choices=["adam", "sgd"])

    parser.add_argument('--loss', type=str, default='wgangp', choices=["wgangp", "lsgan"])

    parser.add_argument('--start_resl', type=float, default=4)
    parser.add_argument('--end_resl',   type=float, default=1024)

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

    os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpus
    torch_device = torch.device("cuda")

    # resolution string will be concatenated in ScalableLoader
    data_path = "../dataset/celeba-"
    
    loader = ScalableLoader(data_path, shuffle=True, drop_last=True, num_workers=arg.cpus, shuffled_cycle=True)

    g = nn.DataParallel(G()).to(torch_device)
    d = nn.DataParallel(D()).to(torch_device)

    tensorboard = utils.TensorboardLogger("%s/tb" % (arg.save_dir))

    model = PGGANrunner(arg, g, d, loader, torch_device, arg.loss, tensorboard)
    with torch.autograd.detect_anomaly():
        model.train()
