import pytest
import torch

import torch.nn as nn

from playground.architectures.FixedConvolutionalImageAutoencoder.decoder import ConvolutionalDecoder


def test_decoder_forward_gray_128():
    batch_size = 32

    image_size = 128
    output_channels = 1
    latent_dim = 10
    
    decoder = ConvolutionalDecoder(
            image_size,
            output_channels,
            latent_dim
        )
    
    z = torch.zeros(batch_size, latent_dim)
    output = decoder(z)

    assert output.shape[0] == batch_size
    assert output.shape[1] == output_channels
    output.shape[2] == image_size
    output.shape[3] == image_size


def test_decoder_forward_color_128():
    batch_size = 64

    image_size = 128
    output_channels = 3
    latent_dim = 10
    
    decoder = ConvolutionalDecoder(
            image_size,
            output_channels,
            latent_dim
        )
    
    z = torch.zeros(batch_size, latent_dim)
    output = decoder(z)

    assert output.shape[0] == batch_size
    assert output.shape[1] == output_channels
    output.shape[2] == image_size
    output.shape[3] == image_size


def test_decoder_forward_gray_56():
    batch_size = 128

    image_size = 56
    output_channels = 1
    latent_dim = 10
    
    decoder = ConvolutionalDecoder(
            image_size,
            output_channels,
            latent_dim
        )
    
    z = torch.zeros(batch_size, latent_dim)
    output = decoder(z)

    assert output.shape[0] == batch_size
    assert output.shape[1] == output_channels
    output.shape[2] == image_size
    output.shape[3] == image_size


def test_decoder_forward_color_56():
    batch_size = 1

    image_size = 56
    output_channels = 3
    latent_dim = 10
    
    decoder = ConvolutionalDecoder(
            image_size,
            output_channels,
            latent_dim
        )
    
    z = torch.zeros(batch_size, latent_dim)
    output = decoder(z)

    assert output.shape[0] == batch_size
    assert output.shape[1] == output_channels
    output.shape[2] == image_size
    output.shape[3] == image_size


def test_decoder_forward_gray_32():
    batch_size = 32

    image_size = 32
    output_channels = 1
    latent_dim = 10
    
    decoder = ConvolutionalDecoder(
            image_size,
            output_channels,
            latent_dim
        )
    
    z = torch.zeros(batch_size, latent_dim)
    output = decoder(z)

    assert output.shape[0] == batch_size
    assert output.shape[1] == output_channels
    output.shape[2] == image_size
    output.shape[3] == image_size


def test_decoder_forward_color_32():
    batch_size = 32

    image_size = 32
    output_channels = 3
    latent_dim = 10
    
    decoder = ConvolutionalDecoder(
            image_size,
            output_channels,
            latent_dim
        )
    
    z = torch.zeros(batch_size, latent_dim)
    output = decoder(z)

    assert output.shape[0] == batch_size
    assert output.shape[1] == output_channels
    output.shape[2] == image_size
    output.shape[3] == image_size
