import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from typing import Tuple

from core import Evaluator


class PerplexityEvaluator(Evaluator):
    def __init__(self, image_model, text_model, dataset, dataset_tokenizer, device='cuda'):
        self.image_model = image_model.to(device)
        self.text_model = text_model.to(device)
        self.dataset = dataset
        self.dataset_tokenizer = dataset_tokenizer
        self.device = device

        self.evaluator_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        if self.evaluator_tokenizer.pad_token is None:
            self.evaluator_tokenizer.pad_token = self.evaluator_tokenizer.eos_token
        self.evaluator_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        self.evaluator_model.eval()

    def evaluate(self) -> Tuple[float, float]:
        return (
            self.__evaluate_image_to_text(),
            self.__evaluate_text_reconstruction()
        )

    def __evaluate_image_to_text(self) -> float:
        self.image_model.eval()
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for img, _ in self.dataset:
                if img.dim() == 2:
                    img = img.unsqueeze(0)
                img = img.to(self.device)

                _, mu, sigma = self.image_model(img)
                z = self.image_model.reparametrize(mu, sigma)
                decoded_logits = self.text_model.decode(z)
                decoded_ids = torch.argmax(decoded_logits, dim=2)

                texts = [self.__list_tokens_to_string(self.dataset_tokenizer.decode(seq)) for seq in decoded_ids.cpu().numpy()]
                inputs = self.evaluator_tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)

                outputs = self.evaluator_model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss

                attention_mask = inputs["attention_mask"]
                batch_tokens = attention_mask.sum().item()

                total_loss += loss.item() * batch_tokens
                total_tokens += batch_tokens

        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        return perplexity

    def __evaluate_text_reconstruction(self) -> float:
        self.text_model.eval()
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for _, texts in self.dataset:
                decoded_texts = []
                for t in texts:
                    if isinstance(t, torch.Tensor):
                        t = t.cpu().numpy()
                    tokens_list = self.dataset_tokenizer.decode(t)
                    text_str = self.__list_tokens_to_string(tokens_list)
                    decoded_texts.append(text_str)

                encoded = [self.dataset_tokenizer.encode(text) for text in decoded_texts]
                max_len = max(len(seq) for seq in encoded)
                padded = [seq + [0] * (max_len - len(seq)) for seq in encoded]
                input_ids = torch.tensor(padded, device=self.device)

                _, mu, sigma = self.text_model(input_ids)
                z = self.text_model.reparametrize(mu, sigma)
                decoded_logits = self.text_model.decode(z)
                decoded_ids = torch.argmax(decoded_logits, dim=2)

                texts_decoded = [self.__list_tokens_to_string(self.dataset_tokenizer.decode(t.cpu().numpy())) for t in decoded_ids]
                inputs = self.evaluator_tokenizer(texts_decoded, return_tensors="pt", padding=True, truncation=True).to(self.device)

                outputs = self.evaluator_model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss

                attention_mask = inputs["attention_mask"]
                batch_tokens = attention_mask.sum().item()

                total_loss += loss.item() * batch_tokens
                total_tokens += batch_tokens

        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        return perplexity

    def __list_tokens_to_string(self, tokens_list):
        return " ".join(tokens_list)