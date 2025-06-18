import torch
import torch.nn as nn
from xlstm import xLSTMBlockStack, xLSTMBlockStackConfig

from core.vae import Encoder
from core import Tensor, Mu, Sigma

from typing import Tuple


class Bidirectional(nn.Module):
    def __init__(self, cfg: xLSTMBlockStackConfig):
        super().__init__()
        self.forward_xlstm = xLSTMBlockStack(cfg)
        self.backward_xlstm = xLSTMBlockStack(cfg)
        
    def forward(self, x: Tensor) -> torch.Tensor:
        future_forward = torch.jit.fork(self.forward_xlstm, x)
        
        x_reversed = torch.flip(x, dims=[1])
        future_backward = torch.jit.fork(self.backward_xlstm, x_reversed)
        
        out_forward = torch.jit.wait(future_forward)
        out_backward = torch.jit.wait(future_backward)
        
        h_forward = out_forward[:, -1]
        h_backward = out_backward[:, -1]
        
        return torch.cat([h_forward, h_backward], dim=-1)
    

class BidirectionalEncoder(Encoder, nn.Module):
    def __init__(self, cfg: xLSTMBlockStackConfig, 
                 latent_dim: int):
        
        super().__init__()
        self.xlstm = Bidirectional(cfg)
        self.mu_proj = nn.Sequential(
            nn.Linear(cfg.embedding_dim * 2, latent_dim)
        )
        
        self.logvar_proj = nn.Sequential(
            nn.Linear(cfg.embedding_dim * 2, latent_dim)
        )

    def forward(self, x: Tensor) -> Tuple[Mu, Sigma]:
        x = self.xlstm(x)
        mu = self.mu_proj(x)
        logvar = self.logvar_proj(x)
        return mu, logvar