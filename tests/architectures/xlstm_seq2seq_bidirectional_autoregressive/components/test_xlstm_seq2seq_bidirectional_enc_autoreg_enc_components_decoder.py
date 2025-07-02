import pytest
import torch
import torch.nn as nn

from playground.architectures.xLSTMSeq2seqBidirectionalAutoregressive.decoder import AutoregressiveDecoder
from omegaconf import OmegaConf
from dacite import from_dict, Config as DaciteConfig
from xlstm import xLSTMBlockStackConfig


PATH = "tests/architectures/xlstm_seq2seq_bidirectional_autoregressive/"

@pytest.fixture
def xlstm_config():
    with open(PATH + "configs/decoder.yml", "r") as file:
        cfg = file.read()

    cfg = OmegaConf.create(cfg)
    return from_dict(data_class=xLSTMBlockStackConfig, data=OmegaConf.to_container(cfg), config=DaciteConfig(strict=True))


def test_autoregressive_forward_dims(xlstm_config):
    batch_size = 32
    vocab_size = 20
    latent_dim = 10

    embedding = nn.Embedding(vocab_size, xlstm_config.embedding_dim//2)
    decoder = AutoregressiveDecoder(xlstm_config, vocab_size, embedding, latent_dim)

    x = torch.zeros(batch_size, latent_dim)
    x = decoder(x)

    assert x.shape[0] == batch_size
    assert x.shape[1] == xlstm_config.context_length
    assert x.shape[2] == vocab_size


def test_autoregressive_step_dims(xlstm_config):
    batch_size = 32
    vocab_size = 20
    latent_dim = 10

    embedding = nn.Embedding(vocab_size, xlstm_config.embedding_dim//2)
    decoder = AutoregressiveDecoder(xlstm_config, vocab_size, embedding, latent_dim)

    x = torch.zeros(batch_size, xlstm_config.embedding_dim//2)
    x, state = decoder.step(x)

    assert x.shape[0] == batch_size
    assert x.shape[1] == 1
    assert x.shape[2] == vocab_size

    assert type(state) == dict