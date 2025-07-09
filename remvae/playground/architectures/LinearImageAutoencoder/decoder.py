from core import LatentTensor, Tensor
from core import Decoder

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearDecoder(Decoder, nn.Module):
    def __init__(self, 
                 image_size: int,
                 output_channels: int,
                 dims: list[int],
                 latent_dim: int):

        super(LinearDecoder, self).__init__()

        self.output_channels = output_channels
        self.image_size = image_size

        self.latent_proj = nn.Linear(latent_dim, dims[0])
        
        output_dim = output_channels * image_size * image_size
        dims.append(output_dim)
        
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))

        self.hidden_layers = nn.ModuleList(layers)

    def forward(self, x: LatentTensor) -> Tensor:
        h = self.latent_proj(x)
        for i, layer in enumerate(self.hidden_layers):
            h = layer(h)
            h = F.relu(h) if i < len(self.hidden_layers) - 1 else F.sigmoid(h)
        h = h.reshape((-1, self.output_channels, self.image_size, self.image_size))
        return h