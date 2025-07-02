import torch
import torch.nn as nn

from core.vae import Encoder
from core import Tensor, Mu, Sigma

from typing import Tuple


class BidirectionalEncoder(Encoder, nn.Module):
    def __init__(self, 
                 embedding_dim: int, 
                 hidden_dim: int,
                 latent_dim: int, 
                 num_layers: int
            ):
        
        super(BidirectionalEncoder, self).__init__()

        self.encoder = nn.GRU(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True
            )
        
        self.mu_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, latent_dim)
        )

        self.sigma_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, latent_dim)
        )

    def forward(self, x: Tensor) -> Tuple[Mu, Sigma]:
        _, h = self.encoder(x)
        h = torch.cat([h[-2], h[-1]], dim=-1)

        mu = self.mu_proj(h)
        sigma = self.sigma_proj(h)

        return mu, sigma
