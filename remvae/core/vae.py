from abc import ABC, abstractmethod
from typing import Tuple

from core import Tensor, Mu, Sigma, LatentTensor

class VariationalAutoencoder(ABC):
    @abstractmethod
    def encode(self, x: Tensor, **kwargs) -> Tuple[Mu, Sigma]:
        pass

    @abstractmethod
    def reparametrize(self, mu: Mu, sigma: Sigma, **kwargs) -> LatentTensor:
        pass

    @abstractmethod
    def decode(self, z: LatentTensor, **kwargs) -> Tensor:
        pass
    

class Encoder(ABC):
    @abstractmethod
    def forward(self, x: Tensor, **kwargs) -> Tuple[Mu, Sigma]:
        pass


class Decoder(ABC):
    @abstractmethod
    def forward(self, z: LatentTensor, **kwargs) -> Tensor:
        pass