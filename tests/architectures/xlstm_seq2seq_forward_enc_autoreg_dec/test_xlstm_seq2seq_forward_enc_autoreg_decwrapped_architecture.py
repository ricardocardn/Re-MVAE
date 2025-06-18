import pytest

import torch
import torch.nn as nn

from playground.architectures.xlstm_seq2seq_forward_enc_autoreg_dec import (
    Builder,
    Wrapper
)

from omegaconf import OmegaConf
from dacite import from_dict, Config as DaciteConfig
from xlstm import xLSTMBlockStackConfig


PATH = "tests/architectures/xlstm_seq2seq_bidirectional_enc_autoreg_enc/"

@pytest.fixture
def encoder_xlstm_config():
    with open(PATH + "configs/encoder.yml", "r") as file:
        cfg = file.read()

    cfg = OmegaConf.create(cfg)
    return from_dict(data_class=xLSTMBlockStackConfig, data=OmegaConf.to_container(cfg), config=DaciteConfig(strict=True))

@pytest.fixture
def decoder_xlstm_config():
    with open(PATH + "configs/decoder.yml", "r") as file:
        cfg = file.read()

    cfg = OmegaConf.create(cfg)
    return from_dict(data_class=xLSTMBlockStackConfig, data=OmegaConf.to_container(cfg), config=DaciteConfig(strict=True))


def test_bidirectional_dims(encoder_xlstm_config, decoder_xlstm_config):
    batch_size = 32
    sample_sentence_length = 15
    vocab_size = 20
    latent_dim = 10

    model = Builder().build(vocab_size, latent_dim, encoder_xlstm_config, decoder_xlstm_config)
    wrapper = Wrapper(model)

    x = torch.zeros(batch_size, sample_sentence_length).long()
    h, mu, sigma = wrapper(x)

    assert h.shape[0] == batch_size
    assert h.shape[1] == decoder_xlstm_config.context_length
    assert h.shape[2] == vocab_size