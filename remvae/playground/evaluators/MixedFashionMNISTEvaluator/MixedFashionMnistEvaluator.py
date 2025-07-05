import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .evaluation_model.simple_cnn import SimpleCNN
from typing import Tuple, List

from core import Evaluator


class FashionMNISTEvaluator(Evaluator):
    def __init__(self, image_model, text_model, dataset, tokenizer, device='cuda', evaluator_path="../../models/evaluators/fashion_mnist_cnn.pth"):
        self.image_model = image_model.to(device)
        self.text_model = text_model.to(device)
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device
        self.evaluator = SimpleCNN().to(device)
        self.evaluator.load_state_dict(torch.load(evaluator_path, map_location=device))
        self.fashion_items = [
            "t-shirt", "trouser", "pullover", "dress", "coat",
            "sandal", "shirt", "sneaker", "bag", "ankle boot"
        ]
    
    def evaluate(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        return (
            self.__evaluate_image_to_text(),
            self.__evaluate_text_to_image()
        )

    def __evaluate_text_to_image(self) -> Tuple[float, float, float]:
        self.image_model.eval()
        gen_imgs = []

        with torch.no_grad():
            for _, desc in self.dataset:
                _, mu, sigma = self.text_model(desc)
                z = self.text_model.reparametrize(mu, sigma)
                decoded_imgs = self.image_model.decode(z)
                gen_imgs.append((decoded_imgs, desc))

        images, targets = self.__prepare_images(gen_imgs)
        images = torch.stack(images, dim=0).to(self.device)
        output = self.evaluator(images)
        predictions = self.__get_predictions(output, topk=3)

        return self.__evaluate_accuracy_topk(predictions, targets)

    def __evaluate_image_to_text(self) -> Tuple[float, float, float]:
        self.image_model.eval()
        gen_imgs = []

        with torch.no_grad():
            for img, _ in self.dataset:
                if img.dim() == 2:
                    img = img.unsqueeze(0)

                img = img.to(self.device)

                _, mu, sigma = self.image_model(img)
                z = self.image_model.reparametrize(mu, sigma)
                decoded_decs = self.text_model.decode(z)
                decoded_decs = torch.argmax(decoded_decs, dim=2)
                gen_imgs.append((img.cpu(), decoded_decs.cpu()))

        images, targets = self.__prepare_images(gen_imgs)
        images = torch.stack(images, dim=0).to(self.device)
        output = self.evaluator(images)
        predictions = self.__get_predictions(output, topk=3)

        return self.__evaluate_accuracy_topk(predictions, targets)

    def __prepare_images(self, data_batches):
        images = []
        targets = []

        for batch_imgs, batch_descs in data_batches:
            for img, desc in zip(batch_imgs, batch_descs):
                if isinstance(desc, torch.Tensor):
                    desc = self.__decode_tensor_to_text(desc)

                if img.dim() == 2:
                    img = img.unsqueeze(0)

                if img.shape[1] != 28 or img.shape[2] != 28:
                    img = F.interpolate(img.unsqueeze(0), size=(28, 28), mode='bilinear', align_corners=False)
                    img = img.squeeze(0)

                images.append(img.cpu())
                targets.append(self.__get_target_from_desc(desc))

        return images, targets

    def __decode_tensor_to_text(self, token_tensor):
        tokens = token_tensor.numpy().tolist()
        if isinstance(tokens[0], list):
            tokens = tokens[0]
        return ' '.join(self.tokenizer.decode(tokens))

    def __get_target_from_desc(self, desc: str):
        for item in self.fashion_items:
            if item in desc:
                return item
        return -1

    def __get_predictions(self, outputs: torch.Tensor, topk: int = 3) -> List[List[str]]:
        topk_indices = torch.topk(outputs, k=topk, dim=1).indices.cpu().numpy()
        return [[self.fashion_items[i] for i in row] for row in topk_indices]

    def __evaluate_accuracy_topk(self, predictions: List[List[str]], targets: List[str]) -> Tuple[float, float, float]:
        top1 = sum(t == p[0] for p, t in zip(predictions, targets))
        top2 = sum(t in p[:2] for p, t in zip(predictions, targets))
        top3 = sum(t in p[:3] for p, t in zip(predictions, targets))
        total = len(targets)
        return top1 / total, top2 / total, top3 / total