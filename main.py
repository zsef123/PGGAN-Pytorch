import os
import argparse

import torch
import torch.nn as nn

from datas.ScalableLoader import ScalableLoader

from models.Generator import G
from models.Discriminator import D
from runners.PGGANrunner import PGGANrunner

import utils


def arg_parse():
    desc = "PGGAN"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--gpus', type=str, default="0,1,2,3",
                        help="Select GPU Numbering | 0,1,2,3 | ")
    parser.add_argument('--cpus', type=int, default="48",
                        help="The number of CPU workers")

    parser.add_argument('--save_dir', type=str, default='',
                        help='Directory which models will be saved in')
    parser.add_argument('--data', type=str, default='celeba-HQ', choices=["celeba-HQ", "RI3d", "RI2d", "RI2d_patch"],
                        help="select dataset")
    parser.add_argument('--load_num', type=int, default=8,
                        help='The number of data files to be loaded to memory at once')
    parser.add_argument('--img_num', type=int, default=600000,
                        help='The number of images to be used for each phase')
    parser.add_argument('--d_iter', type=int, default=1,
                        help='The number of iteration of Discriminator (per 1 step of Generator)')
    parser.add_argument('--cycle_repeat', type=int, default=1,
                        help='The number of iteration of Discriminator (per 1 step of Generator)')

    parser.add_argument('--label_smoothing', type=float, default=0,
                        help='one-sided label smoothing(subtracted only to real label)')
    parser.add_argument('--input_normalize', action="store_true", help="normalize input range to [0, 1]")

    parser.add_argument('--optim_G', type=str, default='adam', choices=["adam", "sgd", "rmsprop"])
    parser.add_argument('--optim_D', type=str, default='adam', choices=["adam", "sgd", "rmsprop"])

    parser.add_argument('--loss', type=str, default='wgangp', choices=["wgangp", "lsgan"])

    parser.add_argument('--start_resl', type=float, default=4)
    parser.add_argument('--end_resl', type=float, default=1024)

    # Adam Optimizer
    parser.add_argument('--beta', nargs="*", type=float, default=(0.0, 0.999),
                        help='Beta for Adam optimizer')
    # SGD Optimizer
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD optimizer')
    parser.add_argument('--decay', type=float, default=0,
                        help='Weight decay for optimizers')
    # Gradient Panelty
    parser.add_argument('--gp_lambda', type=float, default=10.0,
                        help='Lambda as a weight of Gradient Panelty in WGAN-GP loss')
    
    parser.add_argument('--eps_drift', type=float, default=0.001,
                        help='Lambda as a drift term (Appendix A.1)')
                        
    parser.add_argument('--ema_decay', type=float, default=0.999,
                        help='Exponential Moving Average Decay Term')
    # for additional training   
    parser.add_argument('--extra_training_img_num', type=int, default=0,
                        help='The number of images to be used for additional training')
    
    return parser.parse_args()


if __name__ == "__main__":
    arg = arg_parse()

    arg.save_dir = "%s/outs/%s" % (os.getcwd(), arg.save_dir)
    if os.path.exists(arg.save_dir) is False:
        os.mkdir(arg.save_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpus
    torch_device = torch.device("cuda")

    data_path = "../dataset/celeba-"        # resolution string will be concatenated in ScalableLoader
    loader = ScalableLoader(data_path, shuffle=True, drop_last=True, num_workers=arg.cpus, shuffled_cycle=True)

    g = nn.DataParallel(G()).to(torch_device)
    d = nn.DataParallel(D()).to(torch_device)

    tensorboard = utils.TensorboardLogger("%s/tb" % (arg.save_dir))

    model = PGGANrunner(arg, g, d, loader, torch_device, arg.loss, tensorboard)
    model.train()
