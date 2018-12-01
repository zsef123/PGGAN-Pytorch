import random
import torch
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from config import resl_to_batch

class ScalableLoader:
    def __init__(self, path, shuffle=True, drop_last=False, num_workers=4, shuffle_on_cycle=True):
        self.path = path
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.shuffle_on_cycle = shuffle_on_cycle
        
    def __call__(self, resl):        
        batch = resl_to_batch[resl]

        transform = transforms.Compose([transforms.Resize(size=(resl, resl)),
                                        transforms.ToTensor()])
        dataset = ImageFolder(root=self.path, transform=transform)
        
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            num_workers=self.num_workers
        )

        cyclic_loader = self._cyclic_generator(loader)
        return cyclic_loader

    def _cyclic_generator(self, loader):
        while True:
            for element in loader:
                yield element
            if self.shuffle_on_cycle:
                random.shuffle(loader.dataset.imgs)
            
            
    
if __name__ == "__main__":
    
    from itertools import cycle
    sl = ScalableLoader("../dataset", shuffle_on_cycle=False)
    loader = sl(4)
    len_loader = 520
    for idx, item in enumerate(loader):
        print(idx, item)
        if idx % len_loader == 0:
            input()