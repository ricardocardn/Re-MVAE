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
        
        self.attribute_names = [
            "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald",
            "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
            "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
            "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
            "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
            "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks",
            "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
            "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"
        ]

        self.attribute_map = {name: name.replace('_', ' ').lower() for name in self.attribute_names}
        
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
        attr_dict = {name: val for name, val in zip(self.attribute_names, attributes)}

        phrases = []

        intro, genre = self.__get_intro(attr_dict)
        qualifiers = []

        pronoun = 'He' if genre == 0 else 'She'
        poss = 'His' if genre == 0 else 'Her'

        if attr_dict["High_Cheekbones"]: qualifiers.append("high cheekbones")
        if attr_dict["Pointy_Nose"]: qualifiers.append("a pointy nose")
        if attr_dict["Oval_Face"]: qualifiers.append("an oval face")
        if attr_dict["Chubby"]: qualifiers.append("a chubby face")
        if attr_dict["Double_Chin"]: qualifiers.append("a double chin")

        if qualifiers:
            intro += " with " + self.__join_with_and(qualifiers)

        phrases.append(intro + ".")

        hair_descriptions = []
        for color in ["Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair"]:
            if attr_dict[color]:
                hair_descriptions.append(self.attribute_map[color])
        for style in ["Straight_Hair", "Wavy_Hair", "Bangs", "Bald"]:
            if attr_dict[style]:
                hair_descriptions.append(self.attribute_map[style])

        if hair_descriptions:
            phrases.append(pronoun + " has " + self.__join_with_and(hair_descriptions) + ".")

        eye_feats = []
        if attr_dict["Bushy_Eyebrows"]: eye_feats.append("bushy eyebrows")
        if attr_dict["Arched_Eyebrows"]: eye_feats.append("arched eyebrows")
        if attr_dict["Narrow_Eyes"]: eye_feats.append("narrow eyes")
        if attr_dict["Eyeglasses"]: eye_feats.append("glasses")

        if eye_feats:
            phrases.append(pronoun + " has " + self.__join_with_and(eye_feats) + ".")

        makeup = []
        if attr_dict["Heavy_Makeup"]: makeup.append("heavy makeup")
        if attr_dict["Wearing_Lipstick"]: makeup.append("lipstick")
        if attr_dict["Wearing_Earrings"]: makeup.append("earrings")
        if attr_dict["Wearing_Necklace"]: makeup.append("a necklace")
        if attr_dict["Wearing_Necktie"]: makeup.append("a necktie")
        if makeup:
            phrases.append(pronoun + " is wearing " + self.__join_with_and(makeup) + ".")

        facial_hair = []
        if attr_dict["Mustache"]: facial_hair.append("a mustache")
        if attr_dict["Goatee"]: facial_hair.append("a goatee")
        if attr_dict["Sideburns"]: facial_hair.append("sideburns")
        if attr_dict["5_o_Clock_Shadow"]: facial_hair.append("5 o'clock shadow")
        if facial_hair:
            phrases.append(pronoun + " has " + self.__join_with_and(facial_hair) + ".")

        if attr_dict["Smiling"]: phrases.append(pronoun + " is smiling.")
        if attr_dict["Mouth_Slightly_Open"]: phrases.append(poss + " mouth is slightly open.")
        if attr_dict["Young"]: phrases.append(pronoun + " looks young.")
        if attr_dict["Pale_Skin"]: phrases.append(pronoun + " has pale skin.")
        if attr_dict["Rosy_Cheeks"]: phrases.append(pronoun + " has rosy cheeks.")

        return " ".join(phrases) + " <eos>"
    
    def __get_intro(self, attr_dict):
        man = ['A man', 'This man', 'A picture of  man']
        woman = ['A woman', 'This woman', 'A picture of  woman']
        idx = random.randint(0, len(man) - 1)
        return (man[idx], 0) if attr_dict["Male"] == 1 else (woman[idx], 1)
    
    def __join_with_and(self, items):
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return ", ".join(items[:-1]) + f", and {items[-1]}"

    def __load_sequences(self, max_samples: int = 5000):
        sequences = []
        for i in range(min(max_samples, len(self.dataset))):
            attributes = self.dataset[i]["attributes"]
            description = self.__transcript_attributes(attributes)
            sequences.append(description)
        return sequences

    def __set_tokenizer(self, tokenizer):
        if tokenizer is None:
            vocab = generate_vocab(self.sequences, split_by_char=' ')
            return TextTokenizer(vocab)
        else:
            return tokenizer