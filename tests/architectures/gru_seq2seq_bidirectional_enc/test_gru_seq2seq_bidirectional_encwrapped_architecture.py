import pytest

from core.architectures import TextVAE

import torch
import torch.nn as nn


from playground.architectures.gru_seq2seq_bidirectional_enc import (
    Builder,
    Wrapper
)


def test_without_teacher_forcing():
    batch_size = 32
    context_length = 30

    vocab_size = 100
    embedding_dim = 64

    hidden_dim = 64
    latent_dim = 10
    num_layers = 1
    
    builder = Builder()
    model = builder.build(vocab_size, 
                          embedding_dim, 
                          hidden_dim, 
                          latent_dim, 
                          context_length, 
                          num_layers)
    
    assert isinstance(model, TextVAE)

    wrapper = Wrapper(model)

    x = torch.zeros([batch_size, context_length]).long()
    output, mu, sigma = wrapper(x)

    assert output.shape[0] == batch_size
    assert output.shape[1] == context_length
    assert output.shape[2] == vocab_size

    assert mu.shape[0] == batch_size
    assert mu.shape[1] == latent_dim

    assert sigma.shape[0] == batch_size
    assert sigma.shape[1] == latent_dim


def test_with_teacher_forcing():
    batch_size = 32
    context_length = 30

    vocab_size = 100
    embedding_dim = 64

    hidden_dim = 64
    latent_dim = 10
    num_layers = 1
    
    builder = Builder()
    model = builder.build(vocab_size, 
                          embedding_dim, 
                          hidden_dim, 
                          latent_dim, 
                          context_length, 
                          num_layers)
    
    assert isinstance(model, TextVAE)

    wrapper = Wrapper(model)

    x = torch.zeros([batch_size, context_length]).long()
    output, mu, sigma = wrapper(x, x, teacher_forcing=0.5)

    assert output.shape[0] == batch_size
    assert output.shape[1] == context_length
    assert output.shape[2] == vocab_size

    assert mu.shape[0] == batch_size
    assert mu.shape[1] == latent_dim

    assert sigma.shape[0] == batch_size
    assert sigma.shape[1] == latent_dim