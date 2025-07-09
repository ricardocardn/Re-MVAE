import pytest

from playground.readers.OperationsDataset.reader import Reader as OperationsReader
from playground.readers.EnglishSentencesTextDataset.reader import Reader as EnglishReader
from core import Tensor


def test_number_ops_dataset():
    dataset = OperationsReader()
    sample = dataset[0]

    assert len(sample) == 71
    assert type(sample) == Tensor


def test_english_sentences_dataset():
    dataset = EnglishReader()
    sample = dataset[0]

    assert type(sample) == Tensor