from typing import Optional, Callable, Tuple
from torch.utils.data import Dataset
from torchvision import datasets
import torch
import random

from playground.helpers.tokenizer import TextTokenizer
from playground.helpers.text_utils import generate_vocab

from core import Tensor


class Reader(Dataset):
    def __init__(self, 
                 train: Optional[bool] = True, 
                 transform: Optional[Callable] = None, 
                 len: Optional[int] = 2000,
                 tokenizer: Optional[TextTokenizer] = None):
        
        self.dataset = datasets.MNIST(
            root='readers/MNISTMixedDataset/data',
            train=train,
            download=True,
            transform=transform
        )

        self.numbers = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        self.descriptions = [
            "%s <eos>",
            "a %s <eos>",
            "image of a %s <eos>",
            "a photo of a %s <eos>",
            "%s in a picture <eos>",
            "a painting of a %s <eos>",
        ]

        self.sequences = self.__load_sequences()
        self.tokenizer = self.__set_tokenizer(tokenizer)

        self.len = len

    def __len__(self):
        return self.len

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        image, label = self.dataset[idx]
        desc = random.choice(self.descriptions) % self.numbers[label]
        tokens = self.tokenizer.encode(desc, split_by_char=' ')
        return image, torch.tensor(tokens)

    def __load_sequences(self):
        sequences = []
        for desc in self.descriptions:
            for number in self.numbers:
                sequences.append(desc % number)
        return sequences
    
    def __set_tokenizer(self, tokenizer):
        if tokenizer == None:
            vocab = generate_vocab(self.sequences, split_by_char=' ')
            return TextTokenizer(vocab)
        else:
            return tokenizer