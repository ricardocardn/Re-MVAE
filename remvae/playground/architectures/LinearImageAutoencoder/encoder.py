import torch.nn as nn
import torch.nn.functional as F

from core.vae import Encoder
from core import Tensor, Mu, Sigma

from typing import Tuple, List


class LinearEncoder(Encoder, nn.Module):
    def __init__(self, 
                 image_size: int,
                 input_channels: int,
                 dims: List[int],
                 latent_dim: int
            ):
        super(LinearEncoder, self).__init__()

        input_dim = input_channels * image_size * image_size

        layers = []
        prev_dim = input_dim
        for dim in dims:
            layers.append(nn.Linear(prev_dim, dim))
            prev_dim = dim

        self.hidden_layers = nn.ModuleList(layers)

        self.mu_proj = nn.Linear(prev_dim, latent_dim)
        self.logvar_proj = nn.Linear(prev_dim, latent_dim)

    def forward(self, x: Tensor) -> Tuple[Mu, Sigma]:
        h = x.view(x.size(0), -1)
        for i, layer in enumerate(self.hidden_layers):
            h = layer(h)
            h = F.relu(h) if i < len(self.hidden_layers) - 1 else F.tanh(h)  # optional: use different activations
        return self.mu_proj(h), self.logvar_proj(h)