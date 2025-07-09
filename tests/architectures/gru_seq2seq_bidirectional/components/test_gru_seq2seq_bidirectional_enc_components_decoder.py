import pytest
import torch

import torch.nn as nn

from playground.architectures.GRUSeq2seqBidirectional.decoder import AutoregressiveDecoder


def test_decoder_forward():
    batch_size = 32
    context_length = 30

    vocab_size = 100
    embedding_dim = 64
    embedding = nn.Embedding(vocab_size, embedding_dim)

    hidden_dim = 64
    latent_dim = 10
    num_layers = 1
    
    decoder = AutoregressiveDecoder(
            embedding, 
            vocab_size, 
            embedding_dim, 
            hidden_dim, 
            latent_dim, 
            context_length, 
            num_layers
        )
    
    z = torch.zeros(batch_size, latent_dim)
    output = decoder(z)

    assert output.shape[0] == batch_size
    assert output.shape[1] == context_length
    assert output.shape[2] == vocab_size