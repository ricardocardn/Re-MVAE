import pytest
import torch

import torch.nn as nn

from playground.architectures.ConvolutionalNormImageAutoencoder.encoder import ConvolutionalEncoder


def test_decoder_forward_gray_128():
    batch_size = 32

    image_size = 128
    input_channels = 1
    latent_dim = 10
    dims = [1024, 512, 128]
    
    encoder = ConvolutionalEncoder(
            image_size,
            input_channels,
            dims,
            latent_dim
        )
    
    z = torch.zeros(batch_size, input_channels, image_size, image_size)
    mu, sigma = encoder(z)

    assert mu.shape[0] == batch_size
    assert mu.shape[1] == latent_dim

    assert sigma.shape[0] == batch_size
    assert sigma.shape[1] == latent_dim


def test_decoder_forward_color_128():
    batch_size = 64

    image_size = 128
    input_channels = 3
    latent_dim = 10
    dims = [1024, 512, 128]
    
    encoder = ConvolutionalEncoder(
            image_size,
            input_channels,
            dims,
            latent_dim
        )
    
    z = torch.zeros(batch_size, input_channels, image_size, image_size)
    mu, sigma = encoder(z)

    assert mu.shape[0] == batch_size
    assert mu.shape[1] == latent_dim

    assert sigma.shape[0] == batch_size
    assert sigma.shape[1] == latent_dim


def test_decoder_forward_gray_56():
    batch_size = 32

    image_size = 56
    input_channels = 1
    latent_dim = 10
    dims = [1024, 512, 128]
    
    encoder = ConvolutionalEncoder(
            image_size,
            input_channels,
            dims,
            latent_dim
        )
    
    z = torch.zeros(batch_size, input_channels, image_size, image_size)
    mu, sigma = encoder(z)

    assert mu.shape[0] == batch_size
    assert mu.shape[1] == latent_dim

    assert sigma.shape[0] == batch_size
    assert sigma.shape[1] == latent_dim


def test_decoder_forward_color_56():
    batch_size = 64

    image_size = 56
    input_channels = 3
    latent_dim = 10
    dims = [256, 512, 128]
    
    encoder = ConvolutionalEncoder(
            image_size,
            input_channels,
            dims,
            latent_dim
        )
    
    z = torch.zeros(batch_size, input_channels, image_size, image_size)
    mu, sigma = encoder(z)

    assert mu.shape[0] == batch_size
    assert mu.shape[1] == latent_dim

    assert sigma.shape[0] == batch_size
    assert sigma.shape[1] == latent_dim


def test_decoder_forward_gray_32():
    batch_size = 32

    image_size = 32
    input_channels = 1
    latent_dim = 10
    dims = [128, 512, 128]
    
    encoder = ConvolutionalEncoder(
            image_size,
            input_channels,
            dims,
            latent_dim
        )
    
    z = torch.zeros(batch_size, input_channels, image_size, image_size)
    mu, sigma = encoder(z)

    assert mu.shape[0] == batch_size
    assert mu.shape[1] == latent_dim

    assert sigma.shape[0] == batch_size
    assert sigma.shape[1] == latent_dim


def test_decoder_forward_color_32():
    batch_size = 64

    image_size = 32
    input_channels = 3
    latent_dim = 10
    dims = [1024, 512, 256]
    
    encoder = ConvolutionalEncoder(
            image_size,
            input_channels,
            dims,
            latent_dim
        )
    
    z = torch.zeros(batch_size, input_channels, image_size, image_size)
    mu, sigma = encoder(z)

    assert mu.shape[0] == batch_size
    assert mu.shape[1] == latent_dim

    assert sigma.shape[0] == batch_size
    assert sigma.shape[1] == latent_dim