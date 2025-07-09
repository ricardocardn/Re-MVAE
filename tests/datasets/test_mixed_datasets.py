import pytest
from torchvision import transforms

from playground.readers.CelebAMixedDataset.reader import Reader as CelebAMixedReader
from playground.readers.CelebAMixedLargeDataset.reader import Reader as CelebAMixedLargeReader
from playground.readers.FashionMNISTMixedDataset.reader import Reader as FashionMNISTMixedReader
from playground.readers.MNISTMixedDataset.reader import Reader as MNISTMixedReader

from core import Tensor


@pytest.mark.parametrize("ReaderClass", [
    CelebAMixedReader,
    CelebAMixedLargeReader,
    FashionMNISTMixedReader,
    MNISTMixedReader,
])
def test_mixed_datasets_return_image_and_text_tensor(ReaderClass):
    image_size = 64
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    dataset = ReaderClass(transform=transform)
    image_tensor, text_tensor = dataset[0]

    assert isinstance(image_tensor, Tensor), f"{ReaderClass.__name__} did not return a valid image Tensor."
    assert isinstance(text_tensor, Tensor), f"{ReaderClass.__name__} did not return a valid text Tensor."

    assert image_tensor.dim() == 3, f"{ReaderClass.__name__}: image tensor must have 3 dimensions (C, H, W), got {image_tensor.dim()}"
    assert text_tensor.dim() == 1, f"{ReaderClass.__name__}: text tensor must be 1D (token ids), got {text_tensor.dim()}"