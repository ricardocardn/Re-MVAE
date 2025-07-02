from core import LatentTensor, Tensor
from core import Decoder
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalDecoder(Decoder, nn.Module):
    def __init__(self, 
                 image_size: int,
                 output_channels: int,
                 conv_dims: list[int],
                 latent_dim: int):
        
        super(ConvolutionalDecoder, self).__init__()

        self.conv_input_hw = image_size // (2 ** len(conv_dims))
        self.latent_proj = nn.Linear(latent_dim, conv_dims[0] * self.conv_input_hw ** 2)

        layers = []
        in_channels = conv_dims[0]
        for out_channels in conv_dims[1:]:
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1))
            in_channels = out_channels

        self.convs = nn.ModuleList(layers)
        self.final_conv = nn.ConvTranspose2d(in_channels, output_channels, 3, stride=2, padding=1, output_padding=1)

    def forward(self, z: LatentTensor) -> Tensor:
        h = torch.tanh(self.latent_proj(z))
        h = h.view(z.size(0), -1, self.conv_input_hw, self.conv_input_hw)
        for conv in self.convs:
            h = F.relu(conv(h))
        h = torch.sigmoid(self.final_conv(h))
        return h