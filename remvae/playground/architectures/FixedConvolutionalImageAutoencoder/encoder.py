import torch.nn as nn
import torch.nn.functional as F

from core.vae import Encoder
from core import Tensor, Mu, Sigma

from typing import Tuple


class ConvolutionalEncoder(Encoder, nn.Module):
    def __init__(self, 
                 image_size: int,
                 input_channels: int,
                 latent_dim: int
            ):
        
        super(ConvolutionalEncoder, self).__init__()

        self.fc1 = nn.Conv2d(input_channels, 32, 3, stride=2, padding=1)
        self.fc2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.fc3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)

        self.conv_input_hw = image_size // 8
        self.mu_proj = nn.Linear(128 * self.conv_input_hw ** 2, latent_dim)
        self.logvar_proj = nn.Linear(128 * self.conv_input_hw ** 2, latent_dim)

    def forward(self, x: Tensor) -> Tuple[Mu, Sigma]:
        h = F.tanh(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        self._feature_shape = h.shape[1:]
        h = h.view(h.size(0), -1)
        return self.mu_proj(h), self.logvar_proj(h)
