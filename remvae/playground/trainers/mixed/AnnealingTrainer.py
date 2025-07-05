from core import Trainer, Wrapper
from core.architectures import TextVAE, ImageVAE

from playground.helpers.annealing import (
    linear_kl_annealing_func,
    logistic_kl_annealing_func
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from typing import Union, Tuple, Callable


class MixedAnnealingTrainer(Trainer):
    def __init__(self, textVAE: Union[TextVAE, Wrapper],
                 imageVAE: ImageVAE,
                 text_criteria: nn.Module,
                 image_criteria: nn.Module,
                 optimizer: optim.Optimizer,
                 epochs: int,
                 method: Callable,
                 k: int,
                 x0: int,
                 **kwargs):
        
        self.textVAE = textVAE
        self.imageVAE = imageVAE
        self.text_criteria = text_criteria
        self.image_criteria = image_criteria
        self.optimizer = optimizer
        self.epochs = epochs
        self.method = method
        self.k = k
        self.x0 = x0

    def train(self, dataset: Union[Dataset, DataLoader], **kwargs) -> Tuple[TextVAE, ImageVAE]:
        teacher_forcing = kwargs.get('teacher_forcing', 0)
        return_metrics = kwargs.get('return_metrics', False)

        metrics = {'text_model_loss': [], 'image_model_loss': [], 'text_kd_loss': [], 'image_kd_loss': [], 'align_loss': [], 'weight': []}
        step = 0

        for epoch in range(self.epochs):
            total_loss = 0
            total_text_model_loss = 0
            total_image_model_loss = 0
            total_text_kd_loss = 0
            total_image_kd_loss = 0
            total_align_loss = 0

            for iter, (image, sequence) in enumerate(dataset):
                input_sequence = sequence
                target_sequence = sequence

                input_image = image
                target_image = image

                output_text, mu_text, sigma_text = self.textVAE(input_sequence, target_sequence, teacher_forcing)
                output_image, mu_image, sigma_image = self.imageVAE(input_image)


                text_model_loss, image_model_loss, text_kd_loss, image_kd_loss, align_loss, weight = self.__loss_function(
                    output_text, 
                    target_sequence, 
                    mu_text, 
                    sigma_text, 
                    output_image,
                    target_image,
                    mu_image,
                    sigma_image,
                    step)
                
                loss = image_model_loss + 1e2 * text_model_loss + weight * (align_loss + image_kd_loss + text_kd_loss)

                metrics['text_model_loss'].append(text_model_loss.item())
                metrics['image_model_loss'].append(image_model_loss.item())
                metrics['text_kd_loss'].append(text_kd_loss.item())
                metrics['image_kd_loss'].append(image_kd_loss.item())
                metrics['align_loss'].append(align_loss.item())
                metrics['weight'].append(weight)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_text_model_loss += text_model_loss.item()
                total_image_model_loss += image_model_loss.item()
                total_text_kd_loss += text_kd_loss.item()
                total_image_kd_loss += image_kd_loss.item()
                total_align_loss += align_loss.item()

                step += 1

            print(f"Epoch: {epoch}, "
              f"Total Loss: {total_loss/len(dataset)}, "
              f"Text Model Loss: {total_text_model_loss/len(dataset)}, "
              f"Image Model Loss: {total_image_model_loss/len(dataset)}, "
              f"Text KD Loss: {total_text_kd_loss/len(dataset)}, "
              f"Image KD Loss: {total_image_kd_loss/len(dataset)}, "
              f"Align Loss: {total_align_loss/len(dataset)}",
              f"Weight: {weight}")

        return (self.textVAE, self.imageVAE) if not return_metrics else (self.textVAE, self.imageVAE, metrics)

    def __loss_function(self, 
                        output_text, 
                        target_text, 
                        mu_text, 
                        sigma_text,
                        output_image,
                        target_image,
                        mu_image,
                        sigma_image,
                        step):
        
        text_model_loss = self.text_criteria(output_text.transpose(1, 2), target_text)
        text_kd_loss = - 0.5 * torch.sum(1 + torch.log(sigma_text**2) - mu_text**2 - sigma_text**2)

        image_model_loss = self.image_criteria(output_image, target_image)
        image_kd_loss = - 0.5 * torch.sum(1 + torch.log(sigma_image**2) - mu_image**2 - sigma_image**2)

        align_loss = torch.sum((mu_text - mu_image)**2) + torch.sum((sigma_text - sigma_image)**2)
        weight = self.method(step=step, k=self.k, x0=self.x0)

        return text_model_loss, image_model_loss, text_kd_loss, image_kd_loss, align_loss, weight
