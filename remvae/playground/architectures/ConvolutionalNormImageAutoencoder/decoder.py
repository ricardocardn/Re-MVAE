import torch
import torch.nn as nn
import torch.nn.functional as F
from core import LatentTensor, Tensor
from core import Decoder

class ConvolutionalDecoder(Decoder, nn.Module):
    def __init__(self, 
                 image_size: int,
                 output_channels: int,
                 conv_dims: list[int],
                 latent_dim: int):
        
        super().__init__()

        num_layers = len(conv_dims)
        self.conv_output_hw = image_size // (2 ** num_layers)
        self.hidden_dim = conv_dims[0] * self.conv_output_hw ** 2
        self.latent_proj = nn.Linear(latent_dim, self.hidden_dim)

        layers = []
        in_channels = conv_dims[0]
        for i, out_channels in enumerate(conv_dims[1:]):
            if i >= len(conv_dims[1:]) // 2:
                kernel_size = 5
                padding = 2
            else:
                kernel_size = 3
                padding = 1

            layers.append(nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels, out_channels,
                    kernel_size=kernel_size,
                    stride=2, padding=padding, output_padding=1
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
            in_channels = out_channels

        self.decoder_blocks = nn.Sequential(*layers)
        
        self.final_conv = nn.ConvTranspose2d(
            in_channels, output_channels,
            kernel_size=5, stride=2, padding=2, output_padding=1
        )

    def forward(self, z: LatentTensor) -> Tensor:
        h = self.latent_proj(z)
        h = h.view(z.size(0), -1, self.conv_output_hw, self.conv_output_hw)
        h = self.decoder_blocks(h)
        x_recon = torch.sigmoid(self.final_conv(h))
        return x_recon