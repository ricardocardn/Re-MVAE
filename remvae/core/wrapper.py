from core import Tensor
from abc import ABC, abstractmethod


class Wrapper(ABC):
    @abstractmethod
    def forward(self, x: Tensor, **kwargs) -> Tensor:
        pass