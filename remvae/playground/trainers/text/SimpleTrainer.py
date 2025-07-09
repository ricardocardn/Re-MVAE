from core import Trainer
from core.architectures import TextVAE

import torch
import torch.nn as nn
import torch.optim as optim


class TextSimpleTrainer(Trainer):
    def __init__(self, textVAE: nn.Module,
                 criteria: nn.Module,
                 optimizer: optim.Optimizer,
                 epochs: int,
                 batch_size: int):
        
        self.model = textVAE
        self.criteria = criteria
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self, dataset, **kwargs) -> TextVAE:
        if self.__teacher_forcing(**kwargs):
            return self.__train_with_teacher_forcing(dataset, **kwargs)
        else:
            return self.__train_without_teacher_forcing(dataset)


    def __teacher_forcing(self, **kwargs):
        return kwargs.get("teacher_forcing", False)

    def __train_without_teacher_forcing(self, dataset):
        for epoch in range(self.epochs):
            total_loss = 0
            total_kd_loss = 0
            total_model_loss = 0

            for sequence in dataset:
                input = sequence
                target = sequence

                output, mu, sigma = self.model(input)

                kd_loss =  -0.5 * torch.sum(1 + torch.log(sigma**2) - mu**2 - sigma**2)
                model_loss = self.criteria(output.transpose(1, 2), target)
                loss = model_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_kd_loss += kd_loss
                total_model_loss += model_loss

            print(f"Epoch: {epoch}, Loss: {loss/len(dataset)*self.batch_size}, KD Loss: {total_kd_loss/len(dataset)*self.batch_size}, Model Loss: {total_model_loss.item()/len(dataset)*self.batch_size}")

        return self.model


    def __train_with_teacher_forcing(self, dataset, **kwargs):
        teacher_forcing = kwargs['teacher_forcing']
        kd_weight = kwargs.get("kd_weight", 0)
        return_metrics = kwargs.get("return_metrics", False)

        metrics = {'model_loss': []}

        for epoch in range(self.epochs):
            loss = 0
            for sequence in dataset:
                input = sequence
                target = sequence

                output, mu, sigma = self.model(input, target, teacher_forcing)

                loss = self.criteria(output.transpose(1, 2), target) - kd_weight * 0.5 * torch.sum(1 + torch.log(sigma**2) - mu**2 - sigma**2)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss += loss.item()
                metrics['model_loss'].append(loss.item())


            print(f"Epoch: {epoch}, Loss: {loss/len(dataset)*self.batch_size}")

        return self.model if not return_metrics else self.model, metrics