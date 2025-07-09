from core import LatentTensor, Tensor
from core import Decoder

import torch
import torch.nn as nn

from typing import Optional


class AutoregressiveDecoder(Decoder, nn.Module):
    def __init__(self, 
                 embedding: nn.Embedding,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 latent_dim: int,
                 context_length: int,
                 num_layers: Optional[int] = 1):

        super(AutoregressiveDecoder, self).__init__()
        self.embedding = embedding

        self.proj_h = nn.Linear(latent_dim, hidden_dim)
        self.proj_c = nn.Linear(latent_dim, hidden_dim)

        self.decoder = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=False
        )

        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.context_length = context_length

    def forward(self, 
                z: LatentTensor, **kwargs) -> Tensor:
        
        h = self.proj_h(z).unsqueeze(0)
        c = self.proj_c(z).unsqueeze(0)

        x = None

        outputs = []

        for _ in range(self.context_length):
            output, (h, c) = self.step(h, c, x)

            output_logits = output.squeeze(1)
            outputs.append(output_logits)

            predicted_ids = output_logits.argmax(dim=1)
            x = self.embedding(predicted_ids).unsqueeze(1)

        return torch.stack(outputs, dim=1)
    
    def step(self,
             h: Tensor,
             c: Tensor,
             x: Optional[Tensor] = None):

        if x == None:
            x = h.transpose(0, 1)

        x, (h, c) = self.decoder(x, (h, c))
        return self.fc(x), (h, c)