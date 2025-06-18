import pytest

from core.architectures import ImageVAE

import torch
import torch.nn as nn

from playground.architectures.convolutional_image_autoencoder import Builder


def test_decoder_forward():
    batch_size = 32

    image_size = 128
    input_channels = 3
    latent_dim = 10

    dims = [32, 64, 128]
    
    builder = Builder()
    model = builder.build(
        image_size,
        input_channels,
        latent_dim,
        dims
    )
    
    assert isinstance(model, ImageVAE)

    x = torch.rand([batch_size, input_channels, image_size, image_size])
    output, mu, sigma = model(x)

    assert output.shape[0] == batch_size
    assert output.shape[1] == input_channels
    assert output.shape[2] == image_size
    assert output.shape[3] == image_size

    assert mu.shape[0] == batch_size
    assert mu.shape[1] == latent_dim

    assert sigma.shape[0] == batch_size
    assert sigma.shape[1] == latent_dim