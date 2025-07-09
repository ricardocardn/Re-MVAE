from core import VariationalAutoencoder
from abc import ABC, abstractmethod

from typing import Tuple
from core import Tensor, Mu, Sigma, LatentTensor

from core import Encoder, Decoder

import torch
import torch.nn as nn


class ImageVAE(VariationalAutoencoder, nn.Module):
    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 ):
        
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def encode(self, x: Tensor) -> Tuple[Mu, Sigma]:
        return self.encoder(x)
    
    def reparametrize(self, mu: Mu, logvar: Sigma) -> LatentTensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: LatentTensor) -> Tensor:
        return self.decoder(z)
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Mu, Sigma]:
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        x = self.decode(z)
        return x, mu, logvar