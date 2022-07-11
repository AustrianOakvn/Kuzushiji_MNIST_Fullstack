import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, random_split
from torchvision import datasets, transforms
from base import BaseDataLoader
import numpy as np



def get_data_loaders(data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
    trsfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.FashionMNIST(data_dir, train=training, download=True, transform=trsfm)

    loader_args = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers
    }
    if training:
        # split dataset into train and validation set
        num_total = len(dataset)
        if isinstance(validation_split, int):
            assert validation_split > 0
            assert validation_split < num_total, "validation set size is configured to be larger than entire dataset."
            num_valid = validation_split
        else:
            num_valid = int(num_total * validation_split)
        num_train = num_total - num_valid

        train_dataset, valid_dataset = random_split(dataset, [num_train, num_valid])

        train_sampler, valid_sampler = None, None
        if dist.is_initialized():
            loader_args['shuffle']=False
            train_sampler = DistributedSampler(train_dataset)
            valid_sampler = DistributedSampler(valid_dataset)
        return DataLoader(train_dataset, sampler=train_sampler, **loader_args), \
               DataLoader(valid_dataset, sampler=valid_sampler, **loader_args)
    else:
        return DataLoader(dataset, **loader_args)



class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)




class KMnistDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=False, validation_split=0.0, num_workers=1, training=True) -> None:
        super().__init__()

        data = np.load('data_dir')
        self.data = torch.from_numpy(data).long()

