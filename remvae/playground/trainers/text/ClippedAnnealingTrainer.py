from core import Trainer
from core.architectures import TextVAE

from playground.helpers.annealing import (
    linear_kl_annealing_func,
    logistic_kl_annealing_func
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from typing import Union


class TextClippedAnnealingTrainer(Trainer):
    def __init__(self, textVAE: nn.Module,
                 criteria: nn.Module,
                 optimizer: optim.Optimizer,
                 epochs: int,
                 batch_size: int,
                 method: str,
                 k: int,
                 x0: int):
        
        self.model = textVAE
        self.criteria = criteria
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.method = method
        self.k = k
        self.x0 = x0

    def train(self, dataset: Union[Dataset, DataLoader], **kwargs) -> TextVAE:
        teacher_forcing = kwargs.get('teacher_forcing', 0)
        return_metrics = kwargs.get('return_metrics', False)
        step = 0

        metrics = {'model_loss': [], 'kd_loss': [], 'kd_weight': []}

        for epoch in range(self.epochs):
            total_loss = 0
            total_model_loss = 0
            total_kd_loss = 0

            for iter, sequence in enumerate(dataset):
                input = sequence
                target = sequence

                output, mu, sigma = self.model(input, target, teacher_forcing)
                model_loss, kd_loss, kd_weight = self.__loss_function(output, target, mu, sigma, step)
                loss = 1e2 * model_loss + kd_weight * kd_loss

                metrics['model_loss'].append(model_loss.item())
                metrics['kd_loss'].append(kd_loss.item())
                metrics['kd_weight'].append(kd_weight)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_model_loss += model_loss.item()
                total_kd_loss += kd_loss.item()

                step += 1

            print(f"Epoch: {epoch}, Loss: {total_loss/len(dataset)}, Model Loss: {total_model_loss/len(dataset)}, KD Loss: {total_kd_loss/len(dataset)}, KD Weight: {kd_weight}")

        return self.model if not return_metrics else self.model, metrics

    def __loss_function(self, output, target, mu, sigma, step):
        model_loss = self.criteria(output.transpose(1, 2), target)
        kd_loss = - 0.5 * torch.sum(1 + torch.log(sigma**2) - mu**2 - sigma**2)

        if self.method == 'linear':
            kd_weight = linear_kl_annealing_func(step, self.x0)

        elif self.method == 'logistic':
            kd_weight = logistic_kl_annealing_func(step, self.k, self.x0)

        return model_loss, kd_loss, kd_weight
