from core.architectures import TextVAE

from .encoder import BidirectionalEncoder
from .decoder import AutoregressiveDecoder

from xlstm import xLSTMBlockStackConfig
import torch.nn as nn


class Builder():
    def build(self, vocab_size: int,
              latent_dim: int,
              encoder_config: xLSTMBlockStackConfig,
              decoder_config: xLSTMBlockStackConfig
            ) -> TextVAE:
        
        embedding = nn.Embedding(vocab_size, decoder_config.embedding_dim//2)
        encoder = BidirectionalEncoder(encoder_config, latent_dim)
        decoder = AutoregressiveDecoder(decoder_config, vocab_size, embedding, latent_dim)
        return TextVAE(embedding, encoder, decoder)