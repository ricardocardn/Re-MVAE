import pytest
import torch
from playground.architectures.xlstm_seq2seq_forward_enc_autoreg_dec.encoder import ForwardEncoder
from omegaconf import OmegaConf
from dacite import from_dict, Config as DaciteConfig
from xlstm import xLSTMBlockStackConfig


PATH = "tests/architectures/xlstm_seq2seq_forward_enc_autoreg_dec/"

@pytest.fixture
def xlstm_config():
    with open(PATH + "configs/encoder.yml", "r") as file:
        cfg = file.read()

    cfg = OmegaConf.create(cfg)
    return from_dict(data_class=xLSTMBlockStackConfig, data=OmegaConf.to_container(cfg), config=DaciteConfig(strict=True))


def test_forward_encoder_initialization(xlstm_config):
    encoder = ForwardEncoder(xlstm_config, 10)
    x = torch.zeros(32, 15, 32)
    mu, sigma = encoder(x)

    print(mu.shape)

    assert mu.shape[0] == 32
    assert mu.shape[1] == 10

    assert sigma.shape[0] == 32
    assert sigma.shape[1] == 10