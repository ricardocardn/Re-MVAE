import json
import pickle
import os

from typing import List, Set, Optional


class TextTokenizer:
    def __init__(self, vocab: Set[str]):
        self.vocab = vocab
        self.token2idx = {token: idx for idx, token in enumerate(vocab)}
        self.idx2token = {idx: token for idx, token in enumerate(vocab)}

    def encode(self, text: str, split_by_char=' ') -> List[str]:
        if split_by_char == '':
            tokens = list(text)
        else:
            tokens = text.split(split_by_char)
        return [self.token2idx[token] for token in tokens]

    def decode(self, tokens: List[str]) -> List[str]:
        return [self.idx2token[idx] for idx in tokens]

    @classmethod
    def load(cls, filepath: str):
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except (pickle.UnpicklingError, UnicodeDecodeError):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                return cls(data["vocab"])
            except json.JSONDecodeError:
                raise ValueError(f"File '{filepath}' is neither a valid pickle nor JSON file")

    def save(self, filepath: str, format='auto'):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if format == 'auto':
            format = 'pickle' if filepath.endswith('.pkl') else 'json'
        
        if format == 'json':
            vocab_to_save = list(self.vocab) if isinstance(self.vocab, set) else self.vocab
            with open(filepath, 'w') as f:
                json.dump({"vocab": vocab_to_save}, f)
        elif format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
        else:
            raise ValueError("Unsupported format. Use 'json' or 'pickle'.")