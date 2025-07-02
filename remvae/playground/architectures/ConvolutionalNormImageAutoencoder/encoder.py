import torch
import torch.nn as nn
from core.vae import Encoder
from core import Tensor, Mu, Sigma
from typing import Tuple


class ConvolutionalEncoder(Encoder, nn.Module):
    def __init__(self, image_size, input_channels, conv_dims, latent_dim):
        super().__init__()
        layers = []
        in_channels = input_channels
        n_layers = len(conv_dims)

        for i, out_channels in enumerate(conv_dims):
            kernel_size = 5 if i < n_layers // 2 else 3
            padding = kernel_size // 2

            layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
            in_channels = out_channels

        self.convs = nn.Sequential(*layers)

        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, image_size, image_size)
            flat_dim = self.convs(dummy).view(1, -1).size(1)

        self.mu_proj = nn.Linear(flat_dim, latent_dim)
        self.logvar_proj = nn.Linear(flat_dim, latent_dim)

    def forward(self, x: Tensor) -> Tuple[Mu, Sigma]:
        h = self.convs(x)
        h = h.view(h.size(0), -1)
        mu = self.mu_proj(h)
        logvar = torch.clamp(self.logvar_proj(h), min=-10, max=10)
        return mu, logvar