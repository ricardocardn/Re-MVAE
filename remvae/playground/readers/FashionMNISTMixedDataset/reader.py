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
        
        self.dataset = datasets.FashionMNIST(
            root='readers/FashionMNISTMixedDataset/data',
            train=train,
            download=True,
            transform=transform
        )

        self.labels = [
            "t-shirt", "trouser", "pullover", "dress", "coat",
            "sandal", "shirt", "sneaker", "bag", "ankle boot"
        ]
        self.descriptions = [
            "%s <eos>",
            "a %s <eos>",
            "image of a %s <eos>",
            "a photo of a %s <eos>",
            "%s in a picture <eos>",
            "a clothing item: %s <eos>",
            "a fashion item called %s <eos>",
        ]

        self.sequences = self.__load_sequences()
        self.tokenizer = self.__set_tokenizer(tokenizer)

        self.len = len

    def __len__(self):
        return self.len

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        image, label = self.dataset[idx]
        desc = random.choice(self.descriptions) % self.labels[label]
        tokens = self.tokenizer.encode(desc, split_by_char=' ')
        return image, torch.tensor(tokens)

    def __load_sequences(self):
        sequences = []
        for desc in self.descriptions:
            for label in self.labels:
                sequences.append(desc % label)
        return sequences
    
    def __set_tokenizer(self, tokenizer):
        if tokenizer is None:
            vocab = generate_vocab(self.sequences, split_by_char=' ')
            return TextTokenizer(vocab)
        else:
            return tokenizer