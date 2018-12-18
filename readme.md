# PGGAN

PyTorch implementation of `PROGRESSIVE GROWING OF GANS FOR IMPROVED QUALITY, STABILITY, AND VARIATION`


### [[arxiv]](https://arxiv.org/abs/1710.10196) [[Official TF Project]](https://github.com/tkarras/progressive_growing_of_gans)

Authors : [Jihyeong Yoo](https://github.com/YooJiHyeong), [Daewoong Ahn](https://github.com/zsef123)

<hr>

## How to use:

```
python3 main.py -h

usage: main.py [-h] [--gpus GPUS] [--cpus CPUS] [--save_dir SAVE_DIR]
               [--img_num IMG_NUM] [--optim_G {adam,sgd}]
               [--optim_D {adam,sgd}] [--loss {wgangp,lsgan}]
               [--start_resl START_RESL] [--end_resl END_RESL]
               [--beta [BETA [BETA ...]]] [--momentum MOMENTUM]
               [--decay DECAY] [--gp_lambda GP_LAMBDA]

PGGAN

optional arguments:
  -h, --help            show this help message and exit
  --gpus GPUS           Select GPU Numbering | 0,1,2,3 |
  --cpus CPUS           The number of CPU workers
  --save_dir SAVE_DIR   Directory which models will be saved in
  --img_num IMG_NUM     The number of images to be used for each phase
  --optim_G {adam,sgd}
  --optim_D {adam,sgd}
  --loss {wgangp,lsgan}
  --start_resl START_RESL
  --end_resl END_RESL
  --beta [BETA [BETA ...]]
                        Beta for Adam optimizer
  --momentum MOMENTUM   Momentum for SGD optimizer
  --decay DECAY         Weight decay for optimizers
  --gp_lambda GP_LAMBDA
                        Lambda as a weight of Gradient Panelty in WGAN-GP loss
```

<hr>

### TODO 

 - Evaluation Metric
 - Upload Results

<hr>

Reference:
 - https://github.com/tkarras/progressive_growing_of_gans/

