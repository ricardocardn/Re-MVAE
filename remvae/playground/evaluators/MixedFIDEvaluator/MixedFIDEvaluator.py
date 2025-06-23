import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms import Resize
from torchvision.transforms.functional import to_tensor
import PIL

from typing import Tuple

from core import Evaluator


class FIDEvaluator(Evaluator):
    def __init__(self, image_model, text_model, dataset, device='cuda', image_size=(128, 128)):
        self.image_model = image_model.to(device)
        self.text_model = text_model.to(device)
        self.dataset = dataset
        self.device = device
        self.resize = Resize(image_size)
        self.fid = FrechetInceptionDistance(feature=2048).to(device)

    def evaluate(self) -> Tuple[int, int]:
        return (
            self.__evaluate_reconstruction_from_images(),
            self.__evaluate_reconstruction_from_text()
        )

    def __evaluate_reconstruction_from_images(self) -> int:
        self.image_model.eval()
        real_imgs, gen_imgs = [], []

        with torch.no_grad():
            for i, (img, _) in enumerate(self.dataset):
                img = self._prepare_image(img).to(self.device)
                real_imgs.append(img)

                _, mu, sigma = self.image_model(img)
                z = self.image_model.reparametrize(mu, sigma)
                gen_img = self.image_model.decode(z)
                gen_img = self._prepare_image(gen_img)
                gen_imgs.append(gen_img.to(self.device))

        real_tensor = torch.cat(real_imgs, dim=0)
        gen_tensor = torch.cat(gen_imgs, dim=0)

        self.fid.update(self._to_uint8(real_tensor), real=True)
        self.fid.update(self._to_uint8(gen_tensor), real=False)
        return self.fid.compute().item()
    
    def __evaluate_reconstruction_from_text(self) -> int:
        self.image_model.eval()
        real_imgs, gen_imgs = [], []

        with torch.no_grad():
            for i, (img, desc) in enumerate(self.dataset):
                img = self._prepare_image(img).to(self.device)
                real_imgs.append(img)

                desc = desc.to(self.device)
                _, mu, sigma = self.text_model(desc)
                z = self.text_model.reparametrize(mu, sigma)
                gen_img = self.image_model.decode(z)
                gen_img = self._prepare_image(gen_img)
                gen_imgs.append(gen_img.to(self.device))

        real_tensor = torch.cat(real_imgs, dim=0)
        gen_tensor = torch.cat(gen_imgs, dim=0)

        self.fid.update(self._to_uint8(real_tensor), real=True)
        self.fid.update(self._to_uint8(gen_tensor), real=False)
        return self.fid.compute().item()

    def _prepare_image(self, img):
        if isinstance(img, torch.Tensor) and img.dim() == 3:
            img = self.resize(img)
        elif isinstance(img, PIL.Image.Image):
            img = self.resize(to_tensor(img))
        return img.clamp(0, 1)

    def _to_uint8(self, img_tensor):
        return (img_tensor.clamp(0, 1) * 255).byte()
