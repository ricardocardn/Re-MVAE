from typing import Optional, Callable, Tuple
from torch.utils.data import Dataset
from torchvision import datasets, transforms
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

        self.mnist = datasets.MNIST(
            root='readers/mnist_image_dataset/data',
            train=train,
            download=True,
            transform=transform
        )

        self.svhn = datasets.SVHN(
            root='readers/svhn_image_dataset/data',
            split='train' if train else 'test',
            download=True,
            transform=transform
        )

        self.len = len
        self.numbers = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        self.descriptions = [
            "%s <eos>", "a %s <eos>", "image of a %s <eos>",
            "a photo of a %s <eos>", "%s in a picture <eos>",
            "a painting of a %s <eos>",
        ]

        self.mnist_by_label = self.group_by_label(self.mnist, label_index=1)
        self.svhn_by_label = self.group_by_label(self.svhn, label_index=1, svhn=True)

        self.sequences = self.__load_sequences()
        self.tokenizer = self.__set_tokenizer(tokenizer)

    def group_by_label(self, dataset, label_index=1, svhn=False):
        label_dict = {i: [] for i in range(10)}
        for i in range(len(dataset)):
            item = dataset[i]
            label = item[label_index]
            if svhn and label == 10:
                label = 0
            label_dict[label].append(i)
        return label_dict

    def __len__(self):
        return self.len

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor, Tensor]:
        digit = random.randint(0, 9)

        mnist_idx = random.choice(self.mnist_by_label[digit])
        svhn_idx = random.choice(self.svhn_by_label[digit])

        mnist_img, _ = self.mnist[mnist_idx]
        svhn_img, _ = self.svhn[svhn_idx]

        desc = random.choice(self.descriptions) % self.numbers[digit]
        tokens = self.tokenizer.encode(desc, split_by_char=' ')
        return mnist_img, svhn_img, torch.tensor(tokens)

    def __load_sequences(self):
        sequences = []
        for desc in self.descriptions:
            for number in self.numbers:
                sequences.append(desc % number)
        return sequences

    def __set_tokenizer(self, tokenizer):
        if tokenizer is None:
            vocab = generate_vocab(self.sequences, split_by_char=' ')
            return TextTokenizer(vocab)
        else:
            return tokenizer