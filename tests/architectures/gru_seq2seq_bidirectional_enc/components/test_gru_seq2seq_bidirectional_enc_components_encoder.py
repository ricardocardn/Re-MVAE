import pytest
import torch
from playground.architectures.gru_seq2seq_bidirectional_enc.encoder import BidirectionalEncoder


def test_bidirectional_encoder_initialization():
    batch_size = 1
    sample_sentence_length = 15

    embedding_dim = 64
    hidden_dim = 64
    latent_dim = 10
    num_layers = 1
    
    encoder = BidirectionalEncoder(embedding_dim, hidden_dim, latent_dim, num_layers)
    x = torch.zeros(batch_size, sample_sentence_length, embedding_dim)
    mu, sigma = encoder(x)

    print(mu.shape)

    assert mu.shape[0] == batch_size
    assert mu.shape[1] == latent_dim

    assert sigma.shape[0] == batch_size
    assert sigma.shape[1] == latent_dim