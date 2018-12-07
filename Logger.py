import os
import json
from cycler import cycler
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision


# x axis of plot 
LOG_KEYS = {
    "stabilization":"global_step",
    "transition":"global_step",
    "grow":"global_step"
}

# y axis of plot
# save datas like loss, f1-score, PSNR, SSIM ..
# can multiple datas
LOG_VALUES = {
    "stabilization":["step", "resl", "loss_G", "loss_D", "alpha_G", "alpha_D"],
    "transition":["step", "resl", "loss_G", "loss_D", "alpha_G", "alpha_D"],
    "grow":["resl"]
}

class Logger:
      
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.log_file = save_dir + "/log.txt"
        self.buffers = []

    def will_write(self, line):
        print(line)
        self.buffers.append(line)

    def flush(self):
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write("\n".join(self.buffers))
            f.write("\n")
        self.buffers = []

    def write(self, line):
        self.will_write(line)
        self.flush()

    def log_write(self, learn_type, **values):
        """log write in buffers

        ex ) log_write("train", epoch=1, loss=0.3)

        Parmeters:
            learn_type : it must be train, valid or test
            values : values keys in LOG_VALUES
        """
        for k in values.keys():
            if k not in LOG_VALUES[learn_type] and k != LOG_KEYS[learn_type]:
                raise KeyError("%s Log %s keys not in log"%(learn_type, k))
        log = "[%s] %s"%(learn_type, json.dumps(values))
        self.will_write(log)
        if learn_type != "train":
            self.flush()
        
    def log_parse(self, log_key):
        log_dict = OrderedDict()
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                if len(line) == 1 or not line.startswith("[%s]"%(log_key)): 
                    continue
                # line : ~~
                line = line[line.find("] ") + 2:] # ~~
                line_log = json.loads(line)

                train_log_key = line_log[LOG_KEYS[log_key]]
                line_log.pop(LOG_KEYS[log_key], None)
                log_dict[train_log_key] = line_log

        return log_dict
    
    def save_image(self, G, global_step, resl, step, mode, img_num=5):
        latent_z = torch.randn(img_num, 512, 1, 1).cuda()
        generated_img = G.forward(latent_z, "stabilization")

        for idx, img in enumerate(generated_img):
            if not os.path.exists(self.save_dir + "/fig"):
                os.mkdir(self.save_dir + "/fig")
            file_path = "%s/fig/[%03d]_%s_%05d_%05d_%02d.png"% (self.save_dir, resl, mode, global_step, step, idx)
            torchvision.utils.save_image(img, file_path)

    def log_plot(self, log_key, mode="jupyter", 
                 figsize=(12, 12), title="plot", colors=["C1", "C2"]):
        """Plotting Log graph

        Or, mode is slack then save image and r5tu%05d_rn save path

        Parameters:
            log_kestep, y : train, valid, test
            mode : jupyter or slack
            figsize : argument of plt
            title : plot title
        """
        plt.figure(figsize=figsize)
        plt.title(title)
        plt.legend(LOG_VALUES[log_key], loc="best")
        
        ax = plt.subplot(111)
        colors = plt.cm.nipy_spectral(np.linspace(0.1, 0.9, len(LOG_VALUES[log_key])))
        ax.set_prop_cycle(cycler('color', colors))
        
        log_dict = self.log_parse(log_key)
        x = log_dict.keys()
        for keys in LOG_VALUES[log_key]:
            y = [v[keys] for v in log_dict.values()]
            label = keys + ", max : %f"%(max(y))
            ax.plot(x, y, marker="o", linestyle="solid", label=label)
        ax.legend()

        if mode == "jupyter":
            plt.show()
        elif mode == "slack":
            # TODO : Test
            img_path = "tmp.jpg"
            plt.savefig(img_path)
            return img_path

if __name__ == "__main__":
    logger = Logger("outs/test")
    log_dict = logger.log_parse("train")
    response = ""
    for k, v in list(log_dict.items())[-3:]:
        response += "%d, %s\n"%(k, str(v))
    print(response)
    # logger.log_plot("train", title="Train Loss", mode="slack")
    # logger.log_plot("valid", title="Valid Acc")
    
    
