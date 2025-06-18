from typing import Optional, Callable
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import random

from datasets import load_dataset

from playground.helpers.tokenizer import TextTokenizer
from playground.helpers.text_utils import generate_vocab

from core import Tensor


class Reader(Dataset):
    def __init__(self,
                 train: Optional[bool] = True,
                 transform: Optional[Callable] = None,
                 len: Optional[int] = 2000,
                 tokenizer: Optional[TextTokenizer] = None):
        
        split = "train" if train else "validation"
        ds = load_dataset("eurecom-ds/celeba", split=split)
        
        self.dataset = ds
        self.transform = transform
        self.len = len
        
    def __len__(self):
        return self.len

    def __getitem__(self, idx) -> Tensor:
        example = self.dataset[idx]
        image = example["image"]
        
        if self.transform:
            image = self.transform(image)
        
        return image