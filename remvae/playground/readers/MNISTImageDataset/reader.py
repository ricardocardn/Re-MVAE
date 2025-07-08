from typing import Optional, Callable
from torch.utils.data import Dataset
from torchvision import datasets

from core import Tensor


class Reader(Dataset):
    def __init__(self, train: Optional[bool] = True, transform: Optional[Callable] = None, len: Optional[int] = 2000):
        self.dataset = datasets.MNIST(
            root='readers/MNISTImageDataset/data',
            train=train,
            download=True,
            transform=transform
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tensor:
        image, _ = self.dataset[idx]
        return image