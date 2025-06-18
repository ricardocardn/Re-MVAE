import random

import torch
import torch.nn as nn

from playground.helpers.xlstm_recursive.xlstm import xLSTMBlockStack, xLSTMBlockStackConfig

from core.vae import Decoder
from core import Tensor, LatentTensor

from typing import Optional, Tuple


class AutoregressiveDecoder(Decoder, nn.Module):
    def __init__(self, cfg: xLSTMBlockStackConfig, 
                 vocab_size: int,
                 embedding: nn.Embedding, 
                 latent_dim: int):
        
        super().__init__()
        self.embedding = embedding
        self.proj_h = nn.Linear(latent_dim, cfg.embedding_dim // 2)
        self.xlstm = xLSTMBlockStack(cfg)
        self.context_length = cfg.context_length
        self.fc = nn.Linear(cfg.embedding_dim, vocab_size)

    def forward(self, z: LatentTensor) -> Tensor:
        z_proj = self.proj_h(z)
        outputs = []
        state = None
        x = None

        for i in range(self.context_length):
            output, state = self.step(z_proj, x, state)

            output_logits = output.squeeze(1)
            outputs.append(output_logits)

            predicted_ids = output_logits.argmax(dim=1)
            x = self.embedding(predicted_ids)

        return torch.stack(outputs, dim=1)

    def step(self, z: LatentTensor,
         x: Optional[Tensor] = None,
         state: Optional[dict] = None) -> Tuple[Tensor, dict]:

        if z.dim() == 2:
            z_expanded = z.unsqueeze(1)
        elif z.dim() == 3:
            z_expanded = z
        else:
            raise ValueError(f"Unexpected shape for z: {z.shape}")

        if x is None:
            dec_input = z_expanded.repeat(1, 1, 2)
        else:
            x = x.unsqueeze(1)
            dec_input = torch.cat([x, z_expanded], dim=2)

        output, state = self.xlstm.step(dec_input, state)
        logits = self.fc(output)
        return logits, state
