from abc import ABC, abstractmethod

from torch.utils.data import Dataset, DataLoader
from typing import Union, Tuple


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, **kwargs) -> Tuple[int, ...]:
        pass