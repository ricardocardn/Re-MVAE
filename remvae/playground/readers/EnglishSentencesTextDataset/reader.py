from typing import Optional
import os

import torch
from torch.utils.data import Dataset

from playground.helpers.tokenizer import TextTokenizer
from playground.helpers.text_utils import generate_vocab

from core import Tensor


class Reader(Dataset):
    def __init__(self, tokenizer: Optional[TextTokenizer] = None):
        self.path = os.path.join(os.path.dirname(__file__), 'data', 'sequences.txt')
        self.sequences = self.__load_sequences()
        self.tokenizer = self.__set_tokenizer(tokenizer)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx) -> Tensor:
        sequence = self.sequences[idx]
        tokens = self.tokenizer.encode(sequence, split_by_char=' ')
        return torch.tensor(tokens)

    def __load_sequences(self):
        sequences = ""
        with open(self.path, 'r') as file:
            sequences = file.read()

        return sequences.split('\n')
    
    def __set_tokenizer(self, tokenizer):
        if tokenizer == None:
            vocab = generate_vocab(self.sequences, split_by_char=' ')
            return TextTokenizer(vocab)
        else:
            return tokenizer