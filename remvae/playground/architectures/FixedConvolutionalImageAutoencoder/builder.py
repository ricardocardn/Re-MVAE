from core.architectures import ImageVAE

from .encoder import ConvolutionalEncoder
from .decoder import ConvolutionalDecoder


class Builder():
    def build(self, image_size: int,
              input_channels: int,
              latent_dim: int
            ) -> ImageVAE:
        
        encoder = ConvolutionalEncoder(
            image_size,
            input_channels,
            latent_dim
          )

        decoder = ConvolutionalDecoder(
            image_size,
            input_channels,
            latent_dim
        )
        
        return ImageVAE(encoder, decoder)