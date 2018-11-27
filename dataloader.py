import torch
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from config import resl_to_batch

class ScalableLoader:
    def __init__(self, path, shuffle=True, drop_last=False, num_workers=4):
        self.path = path        
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers
        
    def __call__(self, resl):        
        batch = resl_to_batch[resl]

        transform = transforms.Compose([transforms.Scale(size=(resl, resl)),
                                        transforms.ToTensor()])
        dataset = ImageFolder(root=self.path, transform=transform)
        
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            num_workers=self.num_workers
        )
        return loader
