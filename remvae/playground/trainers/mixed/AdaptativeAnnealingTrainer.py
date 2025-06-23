from core import Trainer, Wrapper
from core.architectures import TextVAE, ImageVAE

from playground.helpers.annealing import (
    linear_kl_annealing_func,
    logistic_kl_annealing_func,
    modified_logistic_kl_annealing_func
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

from typing import Union, Tuple
import os
import time


class MixedAdaptativennealingTrainer(Trainer):
    def __init__(self, textVAE: Union[TextVAE, Wrapper],
                 imageVAE: ImageVAE,
                 text_criteria: nn.Module,
                 image_criteria: nn.Module,
                 optimizer: optim.Optimizer,
                 epochs: int,
                 latent_dim: int,
                 weights: dict,
                 method: str,
                 k: int,
                 x0: int):
        
        self.textVAE = textVAE
        self.imageVAE = imageVAE
        self.text_criteria = text_criteria
        self.image_criteria = image_criteria
        self.optimizer = optimizer
        self.epochs = epochs
        self.latent_dim = latent_dim
        self.weights = weights
        self.method = method
        self.k = k
        self.x0 = x0

    def train(self, dataset: Union[Dataset, DataLoader], device, **kwargs) -> Tuple[TextVAE, ImageVAE]:
        teacher_forcing = kwargs.get('teacher_forcing', 0)
        results_dir = kwargs.get('results_dir', "")
        checkpoint_dir = kwargs.get('checkpoint_dir', "")
        checkpoint_steps = kwargs.get('checkpoint_steps', 0)
        return_metrics = kwargs.get('return_metrics', False)

        metrics = {'text_model_loss': [], 'image_model_loss': [], 'text_kd_loss': [], 'image_kd_loss': [], 'align_loss': [], 'weight': []}
        step = 0

        start = time.time()
        for epoch in range(self.epochs):
            total_loss = 0
            total_text_model_loss = 0
            total_image_model_loss = 0
            total_text_kd_loss = 0
            total_image_kd_loss = 0
            total_align_loss = 0

            for iter, (image, sequence) in enumerate(dataset):
                input_sequence = sequence.to(device)
                target_sequence = sequence.to(device)

                input_image = image.to(device)
                target_image = image.to(device)

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
                
                metrics['text_model_loss'].append(text_model_loss.item())
                metrics['image_model_loss'].append(image_model_loss.item())
                metrics['text_kd_loss'].append(text_kd_loss.item())
                metrics['image_kd_loss'].append(image_kd_loss.item())
                metrics['align_loss'].append(align_loss.item())
                metrics['weight'].append(weight)
                
                loss = self.weights['image_model_loss'] * image_model_loss
                loss += self.weights['text_model_loss'] * text_model_loss
                loss += self.weights['align_loss'] * weight * align_loss
                loss += self.weights['image_kd_loss'] * weight * image_kd_loss
                loss += self.weights['text_kd_loss'] * weight * text_kd_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_text_model_loss += text_model_loss.item()
                total_image_model_loss += image_model_loss.item()
                total_text_kd_loss += text_kd_loss.item()
                total_image_kd_loss += image_kd_loss.item()
                total_align_loss += align_loss.item()

                if step % 60 == 0 and results_dir != "":
                    self.__save_samples(device, results_dir, image, sequence, epoch, iter)

                if checkpoint_dir != "" and checkpoint_steps != 0 and (step + 1) % checkpoint_steps == 0:
                    self.__save_checkpoint(checkpoint_dir, epoch, iter)
                    
                step += 1

            print(f"Epoch: {epoch}, "
              f"Total Loss: {total_loss/len(dataset)}, "
              f"Text Model Loss: {total_text_model_loss/len(dataset)}, "
              f"Image Model Loss: {total_image_model_loss/len(dataset)}, "
              f"Text KD Loss: {total_text_kd_loss/len(dataset)}, "
              f"Image KD Loss: {total_image_kd_loss/len(dataset)}, "
              f"Align Loss: {total_align_loss/len(dataset)}",
              f"Weight: {weight}")
            
        if checkpoint_dir != "":
            self.__save_checkpoint(checkpoint_dir, epoch, iter)

        end = time.time()
        elapsed = end - start

        hours, rem = divmod(elapsed, 3600)
        minutes, seconds = divmod(rem, 60)

        print("Execution time: {:0>2.0f}:{:0>2.0f}:{:05.2f}".format(hours, minutes, seconds))
        return (self.textVAE, self.imageVAE) if not return_metrics else (self.textVAE, self.imageVAE, metrics)

    def __save_checkpoint(self, checkpoint_dir, epoch, iter):
        print(f"Checkpoints created at {checkpoint_dir}/epoch{epoch}_iter{iter}")
        os.makedirs(checkpoint_dir + f'/epoch{epoch}_iter{iter}', exist_ok=True)
        torch.save(self.imageVAE.state_dict(), checkpoint_dir + f"/epoch{epoch}_iter{iter}/image_model.pth")

        if isinstance(self.textVAE, Wrapper):
            torch.save(self.textVAE.model.state_dict(), checkpoint_dir + f"/epoch{epoch}_iter{iter}/language_model.pth")
        else:
            torch.save(self.textVAE.state_dict(), checkpoint_dir + f"/epoch{epoch}_iter{iter}/language_model.pth")


    def __save_samples(self, device, results_dir, images, desc, epoch, iter):
        os.makedirs(results_dir + '/results_rec', exist_ok=True)
        os.makedirs(results_dir + '/results_gen', exist_ok=True)
        os.makedirs(results_dir + '/results_gen_text', exist_ok=True)

        with torch.no_grad():
            self.imageVAE.eval()
            recon_images, _, _ = self.imageVAE(images.to(device))
            save_image((torch.cat([images.to('cpu'), recon_images.to('cpu')]) * 0.5 + 0.5).to(device),
                        f'{results_dir}/results_rec/sample_epoch{epoch+1}_step{iter}.png')
                
            _, mu, sigma = self.textVAE(desc.to(device))
            if isinstance(self.textVAE, Wrapper):
                z = self.textVAE.model.reparametrize(mu, sigma)
            else:
                z = self.textVAE.reparametrize(mu, sigma)

            recon_images = self.imageVAE.decode(z)
            save_image((recon_images.to('cpu') * 0.5 + 0.5).to(device),
                        f'{results_dir}/results_gen_text/sample_epoch{epoch+1}_step{iter}.png')

            sample_noise = torch.rand([images.shape[0], self.latent_dim])
            gen_images = self.imageVAE.decoder(sample_noise.to(device))
            save_image((gen_images * 0.5 + 0.5).to(device),
                        f'{results_dir}/results_gen/sample_epoch{epoch+1}_step{iter}.png')
                
            self.imageVAE.train()

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
        text_kd_loss = - 0.5 * torch.mean(1 + torch.log(sigma_text**2) - mu_text**2 - sigma_text**2)

        image_model_loss = self.image_criteria(output_image, target_image)
        image_kd_loss = - 0.5 * torch.mean(1 + torch.log(sigma_image**2) - mu_image**2 - sigma_image**2)

        align_loss = torch.mean((mu_text - mu_image)**2) + torch.mean((sigma_text - sigma_image)**2)

        if self.method == 'linear':
            weight = linear_kl_annealing_func(step, self.x0)

        elif self.method == 'logistic':
            weight = logistic_kl_annealing_func(step, self.k, self.x0)

        elif self.method == 'modified':
            weight = modified_logistic_kl_annealing_func(step, self.k, self.x0)

        return text_model_loss, image_model_loss, text_kd_loss, image_kd_loss, align_loss, weight
