import pytest
from torchvision import transforms

from playground.readers.CelebAImageDataset.reader import Reader as CelebAReader
from playground.readers.FashionMNISTImageDataset.reader import Reader as FashionMNISTReader
from playground.readers.MNISTImageDataset.reader import Reader as MNISTReader

from core import Tensor


@pytest.mark.parametrize("ReaderClass, expected_len", [
    (CelebAReader, None),
    (FashionMNISTReader, None),
    (MNISTReader, None),
])
def test_datasets_return_tensor_with_correct_shape(ReaderClass, expected_len):
    image_size = 64
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    dataset = ReaderClass(transform=transform)
    sample = dataset[0]

    assert isinstance(sample, Tensor), f"{ReaderClass.__name__} did not return a Tensor."

    if expected_len is not None:
        assert len(sample) == expected_len, (
            f"{ReaderClass.__name__} expected tensor of length {expected_len}, got {len(sample)}"
        )

    assert sample.dim() == 3, f"{ReaderClass.__name__} tensor does not have 3 dimensions. Got {sample.dim()}"