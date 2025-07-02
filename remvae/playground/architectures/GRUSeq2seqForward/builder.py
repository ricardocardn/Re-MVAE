from core.architectures import TextVAE
import torch.nn as nn

from .encoder import BidirectionalEncoder
from .decoder import AutoregressiveDecoder

from typing import Optional


class Builder():
    def build(self, vocab_size: int,
              embedding_dim: int,
              hidden_dim: int,
              latent_dim: int,
              context_length: int,
              num_layers: Optional[int] = 1
            ) -> TextVAE:
        
        embedding = nn.Embedding(vocab_size, embedding_dim)
        encoder = BidirectionalEncoder(
            embedding_dim,
            hidden_dim,
            latent_dim,
            num_layers
          )

        decoder = AutoregressiveDecoder(
            embedding,
            vocab_size,
            embedding_dim,
            hidden_dim,
            latent_dim,
            context_length,
            num_layers
          )
        
        return TextVAE(embedding, encoder, decoder)