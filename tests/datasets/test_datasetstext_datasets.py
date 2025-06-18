import pytest

from playground.readers.operations_text_dataset.reader import Reader as OperationsReader
from playground.readers.english_sentences_text_dataset_1.reader import Reader as EnglishReader1
from playground.readers.english_sentences_text_dataset_1.reader import Reader as EnglishReader2
from core import Tensor


def test_number_ops_dataset():
    dataset = OperationsReader()
    sample = dataset[0]

    assert len(sample) == 71
    assert type(sample) == Tensor


def test_english_sentences_dataset_1():
    dataset = EnglishReader1()
    sample = dataset[0]

    assert type(sample) == Tensor


def test_english_sentences_dataset_2():
    dataset = EnglishReader2()
    sample = dataset[0]

    assert type(sample) == Tensor