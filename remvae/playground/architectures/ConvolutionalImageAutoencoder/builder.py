from core.architectures import ImageVAE
from .encoder import ConvolutionalEncoder
from .decoder import ConvolutionalDecoder

from typing import Optional


class Builder():
    def build(self, image_size: int,
              input_channels: int,
              latent_dim: int,
              conv_dims: Optional[list[int]] = [32, 64, 128]
            ) -> ImageVAE:
        
        encoder = ConvolutionalEncoder(
            image_size,
            input_channels,
            conv_dims,
            latent_dim
        )

        decoder = ConvolutionalDecoder(
            image_size,
            input_channels,
            conv_dims[::-1],
            latent_dim
        )

        return ImageVAE(encoder, decoder)