import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms.functional import to_tensor, to_pil_image
import numpy as np

from transformers import AutoModelForImageClassification, AutoImageProcessor, pipeline

from typing import Tuple


class MNISTEvaluator:
    def __init__(self, image_model, text_model, dataset, tokenizer, device='cuda'):
        self.image_model = image_model.to(device)
        self.text_model = text_model.to(device)
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device
        self.evaluator = self.__load_evaluator()

    def __load_evaluator(self):
        repo_id = "farleyknight/mnist-digit-classification-2022-09-04"
        processor = AutoImageProcessor.from_pretrained(repo_id)
        model = AutoModelForImageClassification.from_pretrained(repo_id)
        classifier = pipeline("image-classification", model=model, image_processor=processor)
        return classifier
    
    def evaluate(self) -> Tuple[int, int]:
        return (
            self.__evaluate_image_to_text(),
            self.__evaluate_text_to_image()
        )

    def __evaluate_text_to_image(self) -> int:
        self.image_model.eval()
        gen_imgs = []

        with torch.no_grad():
            for _, desc in self.dataset:
                _, mu, sigma = self.text_model(desc)
                z = self.text_model.reparametrize(mu, sigma)
                decoded_imgs = self.image_model.decode(z)
                gen_imgs.append((decoded_imgs, desc))

        pil_images, targets = self.__prepare_images(gen_imgs)
        output = self.evaluator(pil_images)
        results = self.__get_predictions(output)

        correct_samples = np.sum(np.array(targets) == np.array(results))
        total_samples = len(results)
        return correct_samples / total_samples
    
    def __evaluate_image_to_text(self) -> int:
        self.image_model.eval()
        gen_imgs = []

        with torch.no_grad():
            for img, _ in self.dataset:
                _, mu, sigma = self.image_model(img)
                z = self.image_model.reparametrize(mu, sigma)
                decoded_decs = self.text_model.decode(z)
                decoded_decs = torch.argmax(decoded_decs, dim=2)
                gen_imgs.append((img, decoded_decs))

        pil_images, targets = self.__prepare_images(gen_imgs)
        output = self.evaluator(pil_images)
        results = self.__get_predictions(output)

        correct_samples = np.sum(np.array(targets) == np.array(results))
        total_samples = len(results)
        return correct_samples / total_samples

    def __prepare_images(self, images):
        pil_images = []
        targets = []

        for batch_imgs, batch_descs in images:
            for img, desc in zip(batch_imgs, batch_descs):
                if isinstance(desc, torch.Tensor):
                    desc = self.__decode_tensor_to_text(desc)
                pil_images.append(to_pil_image(img.cpu()))
                targets.append(self.__get_target_from_desc(desc))

        return pil_images, targets

    def __decode_tensor_to_text(self, token_tensor):
        return ' '.join(self.tokenizer.decode(token_tensor.numpy().tolist()))

    def __get_target_from_desc(self, desc):
        numbers = [
            "zero", "one", "two", "three", "four",
            "five", "six", "seven", "eight", "nine"
        ]
        
        for number in numbers:
            if number in desc:
                return number
            
        return -1
    
    def __get_predictions(self, outputs):
        number_to_text = {
            "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
            "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"
        }
        results = []
        for output in outputs:
            results.append(number_to_text[output[0]["label"]])
        return results