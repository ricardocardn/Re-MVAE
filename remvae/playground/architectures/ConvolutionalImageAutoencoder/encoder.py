import torch.nn as nn
import torch.nn.functional as F

from core.vae import Encoder
from core import Tensor, Mu, Sigma

from typing import Tuple


class ConvolutionalEncoder(Encoder, nn.Module):
    def __init__(self, 
                 image_size: int,
                 input_channels: int,
                 conv_dims: list[int],
                 latent_dim: int):
        
        super(ConvolutionalEncoder, self).__init__()

        layers = []
        in_channels = input_channels
        for out_channels in conv_dims:
            layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1))
            in_channels = out_channels

        self.convs = nn.ModuleList(layers)

        self.conv_input_hw = image_size // (2 ** len(conv_dims))
        flat_dim = conv_dims[-1] * self.conv_input_hw ** 2

        self.mu_proj = nn.Linear(flat_dim, latent_dim)
        self.logvar_proj = nn.Linear(flat_dim, latent_dim)

    def forward(self, x: Tensor) -> Tuple[Mu, Sigma]:
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h)
            h = F.relu(h) if i > 0 else F.tanh(h)
        self._feature_shape = h.shape[1:]
        h = h.view(h.size(0), -1)
        return self.mu_proj(h), self.logvar_proj(h)