import pytest
import torch
from playground.architectures.xLSTMSeq2seqBidirectionalAutoregressive.encoder import Bidirectional, BidirectionalEncoder
from omegaconf import OmegaConf
from dacite import from_dict, Config as DaciteConfig
from xlstm import xLSTMBlockStackConfig


PATH = "tests/architectures/xlstm_seq2seq_bidirectional_autoregressive/"

@pytest.fixture
def xlstm_config():
    with open(PATH + "configs/encoder.yml", "r") as file:
        cfg = file.read()

    cfg = OmegaConf.create(cfg)
    return from_dict(data_class=xLSTMBlockStackConfig, data=OmegaConf.to_container(cfg), config=DaciteConfig(strict=True))

def test_bidirectional_dims(xlstm_config):
    encoder = Bidirectional(xlstm_config)
    x = torch.zeros(32, 15, 32)
    h = encoder(x)

    assert h.shape[0] == 32
    assert h.shape[1] == 2 * xlstm_config.embedding_dim

def test_bidirectional_encoder_initialization(xlstm_config):
    encoder = BidirectionalEncoder(xlstm_config, 10)
    x = torch.zeros(32, 15, 32)
    mu, sigma = encoder(x)

    assert mu.shape[0] == 32
    assert mu.shape[1] == 10

    assert sigma.shape[0] == 32
    assert sigma.shape[1] == 10