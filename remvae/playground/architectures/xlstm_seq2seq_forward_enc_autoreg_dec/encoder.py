import torch
import torch.nn as nn
from xlstm import xLSTMBlockStack, xLSTMBlockStackConfig

from core.vae import Encoder
from core import Tensor, Mu, Sigma

from typing import Tuple
    

class ForwardEncoder(Encoder, nn.Module):
    def __init__(self, cfg: xLSTMBlockStackConfig, 
                 latent_dim: int):
        
        super().__init__()
        self.xlstm = xLSTMBlockStack(cfg)
        self.mu_proj = nn.Sequential(
            nn.Linear(cfg.embedding_dim, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, latent_dim)
        )
        
        self.logvar_proj = nn.Sequential(
            nn.Linear(cfg.embedding_dim, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, latent_dim)
        )

    def forward(self, x: Tensor) -> Tuple[Mu, Sigma]:
        x = self.xlstm(x)[:, -1, :]
        mu = self.mu_proj(x)
        logvar = self.logvar_proj(x)
        return mu, logvar