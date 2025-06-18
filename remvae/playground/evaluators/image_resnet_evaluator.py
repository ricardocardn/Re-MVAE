import torch
import torchvision.models as models
from torchvision.transforms import Resize
from torchvision.transforms.functional import to_tensor
import torch.nn.functional as F
from typing import List
import PIL


class ResnetEmbeddingEvaluator:
    def __init__(self, image_model, dataset, device='cuda', metric='l2'):
        self.image_model = image_model.to(device)
        self.dataset = dataset
        self.device = device
        self.metric = metric
        self.resize = Resize((224, 224))
        self.resnet = self.__load_resnet()

    def __load_resnet(self):
        model = models.resnet18(pretrained=True)
        model.fc = torch.nn.Identity()
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return model.to(self.device)

    def evaluate(self) -> float:
        self.image_model.eval()
        distances: List[float] = []

        with torch.no_grad():
            for img, _ in self.dataset:
                real_img = self.__prepare_image(img).unsqueeze(0).to(self.device)

                gen_img, _, _ = self.image_model(real_img)
                gen_img = self.__prepare_image(gen_img).unsqueeze(0).to(self.device)

                emb_real = self.resnet(real_img)
                emb_gen = self.resnet(gen_img)

                dist = self.__embedding_distance(emb_real, emb_gen)
                distances.append(dist.item())

        return sum(distances) / len(distances)

    def __prepare_image(self, img):
        if isinstance(img, torch.Tensor) and img.dim() == 3:
            img = self.resize(img)
        elif isinstance(img, PIL.Image.Image):
            img = self.resize(to_tensor(img))
        return img.clamp(0, 1)

    def __embedding_distance(self, emb1, emb2):
        if self.metric == 'cosine':
            return 1 - F.cosine_similarity(emb1, emb2).mean()
        elif self.metric == 'l2':
            return F.pairwise_distance(emb1, emb2).mean()
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")