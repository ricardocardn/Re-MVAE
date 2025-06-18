from typing import TypeAlias
import torch

Tensor: TypeAlias = torch.Tensor
Mu: TypeAlias = Tensor
Sigma: TypeAlias = Tensor
LatentTensor: TypeAlias = Tensor

from .vae import (
    VariationalAutoencoder,
    Encoder,
    Decoder,
)

from .trainer import Trainer
from .wrapper import Wrapper