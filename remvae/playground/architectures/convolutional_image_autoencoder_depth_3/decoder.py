from core import LatentTensor, Tensor
from core import Decoder

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalDecoder(Decoder, nn.Module):
    def __init__(self, 
                 image_size: int,
                 output_channels: int,
                 latent_dim: int):

        super(ConvolutionalDecoder, self).__init__()
        
        self.conv_input_hw = image_size // 8
        self.latent_proj = nn.Linear(latent_dim, 128 * self.conv_input_hw ** 2)
        self.fc4 = nn.ConvTranspose2d(128, 64, 3, stride=2, output_padding=1, padding=1)
        self.fc5 = nn.ConvTranspose2d(64, 32, 3, stride=2, output_padding=1, padding=1)
        self.fc6 = nn.ConvTranspose2d(32, output_channels, 3, stride=2, output_padding=1, padding=1)

    def forward(self, 
                z: LatentTensor, **kwargs) -> Tensor:
        
        h = torch.tanh(self.latent_proj(z))
        h = h.view(z.size(0), 128, self.conv_input_hw, self.conv_input_hw)
        h = F.relu(self.fc4(h))
        h = F.relu(self.fc5(h))
        h = torch.sigmoid(self.fc6(h))
        return h