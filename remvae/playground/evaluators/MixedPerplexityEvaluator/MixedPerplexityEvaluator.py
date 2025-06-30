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

    def evaluate(self) -> Tuple[float, float, float]:
        return (
            self.__evaluate_original_texts_perplexity(),
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

                texts = []
                for seq in decoded_ids.cpu().numpy():
                    text = self.__list_tokens_to_string(self.dataset_tokenizer.decode(seq))
                    if len(text.strip().split()) <= 1:
                        continue
                    texts.append(text)

                if not texts:
                    continue

                inputs = self.evaluator_tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
                outputs = self.evaluator_model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss

                attention_mask = inputs["attention_mask"]
                batch_tokens = attention_mask.sum().item()

                if torch.isnan(loss) or batch_tokens < 2:
                    continue

                total_loss += loss.item() * batch_tokens
                total_tokens += batch_tokens

        if total_tokens == 0:
            return float('nan')

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
                    text_str = self.__list_tokens_to_string(self.dataset_tokenizer.decode(t))
                    if len(text_str.strip().split()) <= 1:
                        continue
                    decoded_texts.append(text_str)

                if not decoded_texts:
                    continue

                inputs = self.evaluator_tokenizer(decoded_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
                outputs = self.evaluator_model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss

                attention_mask = inputs["attention_mask"]
                batch_tokens = attention_mask.sum().item()

                if torch.isnan(loss) or batch_tokens < 2:
                    continue

                total_loss += loss.item() * batch_tokens
                total_tokens += batch_tokens

        if total_tokens == 0:
            return float('nan')

        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        return perplexity

    def __evaluate_original_texts_perplexity(self) -> float:
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for _, texts in self.dataset:
                original_texts = []
                for t in texts:
                    if isinstance(t, torch.Tensor):
                        t = t.cpu().numpy()
                    text_str = self.__list_tokens_to_string(self.dataset_tokenizer.decode(t))
                    if len(text_str.strip().split()) <= 1:
                        continue
                    original_texts.append(text_str)

                if not original_texts:
                    continue

                inputs = self.evaluator_tokenizer(original_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
                outputs = self.evaluator_model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss

                attention_mask = inputs["attention_mask"]
                batch_tokens = attention_mask.sum().item()

                if torch.isnan(loss) or batch_tokens < 2:
                    continue

                total_loss += loss.item() * batch_tokens
                total_tokens += batch_tokens

        if total_tokens == 0:
            return float('nan')

        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        return perplexity

    def __list_tokens_to_string(self, tokens_input):
        if isinstance(tokens_input, list):
            text = " ".join(tokens_input)
        else:
            text = tokens_input
        return text.split(' <eos>')[0].strip()