from typing import Optional, Callable, Tuple
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
        
        attribute_names = [
            "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald",
            "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
            "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
            "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
            "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
            "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks",
            "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
            "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"
        ]

        self.attribute_names = [name.replace('_', ' ') for name in attribute_names]
        
        self.sequences = self.__load_sequences()
        self.tokenizer = self.__set_tokenizer(tokenizer)
        
    def __len__(self):
        return self.len

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        example = self.dataset[idx]
        image = example["image"]
        attributes = example["attributes"]
        
        if self.transform:
            image = self.transform(image)
        
        description = self.__transcript_attributes(attributes)
        tokens = self.tokenizer.encode(description, split_by_char=' ')
        
        return image, torch.tensor(tokens)

    def __transcript_attributes(self, attributes):
        active_attributes = []
        count = 5
        for name, value in zip(self.attribute_names, attributes):
            if value == 1 and count > 0:
                active_attributes.append(name)
                count -= 1

            elif count == 0:
                break
            
        if not active_attributes:
            return "no visible attributes <eos>"
        return ", ".join(active_attributes) + " <eos>"
    
    def __load_sequences(self):
        sequences = []
        for name in self.attribute_names:
            sequences.append(f"{name} {name}, <eos>")
        sequences.append("no visible attributes <eos>")
        return sequences

    def __set_tokenizer(self, tokenizer):
        if tokenizer is None:
            vocab = generate_vocab(self.sequences, split_by_char=' ')
            return TextTokenizer(vocab)
        else:
            return tokenizer