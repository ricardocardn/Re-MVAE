from core.architectures import ImageVAE

from .encoder import LinearEncoder
from .decoder import LinearDecoder

from typing import Optional


class Builder():
    def build(self, image_size: int,
              input_channels: int,
              latent_dim: int,
              dims: Optional[list[int]] = [1024, 512, 128]
            ) -> ImageVAE:
        
        encoder = LinearEncoder(
            image_size,
            input_channels,
            dims,
            latent_dim
          )

        decoder = LinearDecoder(
            image_size,
            input_channels,
            dims[::-1],
            latent_dim
        )
        
        return ImageVAE(encoder, decoder)