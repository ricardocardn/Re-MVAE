from typing import List, Set, Optional

def generate_vocab(sentences: List[str], split_by_char: Optional[str] = '') -> Set[str]:
    vocab = set()
    for sentence in sentences:
        if split_by_char == '':
            tokens = list(sentence)
        else:
            tokens = sentence.split(split_by_char)

        tokens = [token for token in tokens if token]
        vocab.update(tokens)

    vocab.add('<pad>')
    return vocab