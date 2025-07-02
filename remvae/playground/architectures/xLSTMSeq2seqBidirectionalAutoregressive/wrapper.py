from core.architectures import TextVAE
from core import Tensor

import torch
import torch.nn as nn

import random

from core.wrapper import Wrapper
from typing import Optional


class xLSTMWrapper(Wrapper, nn.Module):
    def __init__(self, model: TextVAE):
        super().__init__()
        self.model = model

    def forward(self, x: Tensor,
                target: Optional[Tensor] = None,
                teacher_forcing: Optional[int] = 0) -> Tensor:
        
        x = self.model.embedding(x)
        mu, logvar = self.model.encode(x)
        z = self.model.reparametrize(mu, logvar)
        z = self.model.decoder.proj_h(z)

        outputs = []
        
        for i in range(self.model.decoder.context_length):
            if i == 0:
                output, state = self.model.decoder.step(z.unsqueeze(1))
            else:
                output, state = self.model.decoder.step(z, output, state)

            outputs.append(output)

            if not self.teacher_forcing(target, teacher_forcing, i):
                output = output.squeeze(1)
                output = output.argmax(dim=1)
                output = self.model.embedding(output)
            else:
                output = target[:, i]
                output = self.model.embedding(output)

        return torch.cat(outputs, dim=1), mu, logvar

    def teacher_forcing(self, target, teacher_forcing, i):
        return random.random() < teacher_forcing and target != None and i < self.model.decoder.context_length