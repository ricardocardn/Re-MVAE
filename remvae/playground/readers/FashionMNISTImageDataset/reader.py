from typing import Optional, Callable
from torch.utils.data import Dataset
from torchvision import datasets
import torch
import random

from core import Tensor


class Reader(Dataset):
    def __init__(self, 
                 train: Optional[bool] = True, 
                 transform: Optional[Callable] = None):
        
        self.dataset = datasets.FashionMNIST(
            root='readers/FashionMNISTImageDataset/data',
            train=train,
            download=True,
            transform=transform
        )

    def __len__(self):
        return self.len

    def __getitem__(self, idx) -> Tensor:
        image, _ = self.dataset[idx]
        return image