from core import Trainer
from core.architectures import ImageVAE

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

from typing import Union
import os
import time


class ImageClippedTrainer(Trainer):
    def __init__(self, imageVAE: ImageVAE,
                 image_criteria: nn.Module,
                 optimizer: optim.Optimizer,
                 epochs: int,
                 latent_dim: int,
                 method: str,
                 k: int,
                 x0: int):
        
        self.imageVAE = imageVAE
        self.image_criteria = image_criteria
        self.optimizer = optimizer
        self.epochs = epochs
        self.latent_dim = latent_dim
        self.method = method
        self.k = k
        self.x0 = x0

    def train(self, dataset: Union[Dataset, DataLoader], device, kd_min=0.5, **kwargs) -> ImageVAE:
        results_dir = kwargs.get('results_dir', "")
        checkpoint_dir = kwargs.get('checkpoint_dir', "")
        checkpoint_steps = kwargs.get('checkpoint_steps', 0)
        return_metrics = kwargs.get('return_metrics', False)

        metrics = {'image_model_loss': [], 'image_kd_loss': []}
        step = 0

        start = time.time()
        for epoch in range(self.epochs):
            total_loss = 0
            total_image_model_loss = 0
            total_image_kd_loss = 0

            for iter, image in enumerate(dataset):
                input_image = image.to(device)
                target_image = image.to(device)

                output_image, mu, sigma = self.imageVAE(input_image)

                image_model_loss, image_kd_loss = self.__loss_function(
                    output_image,
                    target_image,
                    mu,
                    sigma)
                
                metrics['image_model_loss'].append(image_model_loss.item())
                metrics['image_kd_loss'].append(image_kd_loss.item())
                
                loss = image_model_loss + (kd_min - image_kd_loss)**2

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_image_model_loss += image_model_loss.item()
                total_image_kd_loss += image_kd_loss.item()

                if step % 60 == 0 and results_dir != "":
                    self.__save_samples(device, results_dir, image, epoch, iter)

                if checkpoint_dir != "" and checkpoint_steps != 0 and (step + 1) % checkpoint_steps == 0:
                    self.__save_checkpoint(checkpoint_dir, epoch, iter)
                    
                step += 1

            print(f"Epoch: {epoch}, "
              f"Total Loss: {total_loss/len(dataset)}, "
              f"Image Model Loss: {total_image_model_loss/len(dataset)}, "
              f"Image KD Loss: {total_image_kd_loss/len(dataset)}")
            
        if checkpoint_dir != "":
            self.__save_checkpoint(checkpoint_dir, epoch, iter)

        end = time.time()
        elapsed = end - start

        hours, rem = divmod(elapsed, 3600)
        minutes, seconds = divmod(rem, 60)

        print("Execution time: {:0>2.0f}:{:0>2.0f}:{:05.2f}".format(hours, minutes, seconds))
        return self.imageVAE if not return_metrics else (self.imageVAE, metrics)

    def __save_checkpoint(self, checkpoint_dir, epoch, iter):
        print(f"Checkpoints created at {checkpoint_dir}/epoch{epoch}_iter{iter}")
        os.makedirs(checkpoint_dir + f'/epoch{epoch}_iter{iter}', exist_ok=True)
        torch.save(self.imageVAE.state_dict(), checkpoint_dir + f"/epoch{epoch}_iter{iter}/image_model.pth")


    def __save_samples(self, device, results_dir, images, epoch, iter):
        os.makedirs(results_dir + '/results_rec', exist_ok=True)
        os.makedirs(results_dir + '/results_gen', exist_ok=True)

        with torch.no_grad():
            self.imageVAE.eval()
            recon_images, _, _ = self.imageVAE(images.to(device))
            save_image((torch.cat([images.to('cpu'), recon_images.to('cpu')]) * 0.5 + 0.5).to(device),
                        f'{results_dir}/results_rec/sample_epoch{epoch+1}_step{iter}.png')

            sample_noise = torch.rand([images.shape[0], self.latent_dim])
            gen_images = self.imageVAE.decoder(sample_noise.to(device))
            save_image((gen_images * 0.5 + 0.5).to(device),
                        f'{results_dir}/results_gen/sample_epoch{epoch+1}_step{iter}.png')
                
            self.imageVAE.train()

    def __loss_function(self, 
                        output_image,
                        target_image,
                        mu_image,
                        sigma_image):

        image_model_loss = self.image_criteria(output_image, target_image)
        image_kd_loss = - 0.5 * torch.mean(1 + torch.log(sigma_image**2) - mu_image**2 - sigma_image**2)

        return image_model_loss, image_kd_loss