import os
import json
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from playground.readers.FashionMNISTMixedDataset import Reader
from playground.architectures.FixedConvolutionalImageAutoencoder import Builder as ImageBuilder
from playground.architectures.GRUSeq2seqBidirectional import Builder as TextBuilder, Wrapper
from playground.evaluators import (MixedFashionMNISTEvaluator, MixedPerplexityEvaluator)
from playground.helpers.tokenizer import TextTokenizer



def load_models_and_data(args_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open(args_path, 'r') as f:
        args = json.load(f)

    tokenizer = TextTokenizer.load(args["tokenizer_path"])
    transform = transforms.Compose([
        transforms.Resize((args["image_size"], args["image_size"])),
        transforms.ToTensor()
    ])

    dataset = Reader(train=False, transform=transform, len=args["dataset_length"])
    dataset.tokenizer = tokenizer
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    image_model = ImageBuilder().build(
        args["image_size"], args["input_channels"], args["latent_dim"]
        )

    text_model = TextBuilder().build(
        vocab_size=len(dataset.tokenizer.vocab),
        embedding_dim=args["embedding_dim"],
        hidden_dim=args["hidden_dim"],
        latent_dim=args["latent_dim"],
        context_length=args["context_length"],
        num_layers=1
        )


    wrapper = Wrapper(text_model)

    checkpoint_base = args["checkpoint_dir"]
    dirs = [os.path.join(checkpoint_base, d) for d in os.listdir(checkpoint_base)]
    dirs = [d for d in dirs if os.path.isdir(d)]

    checkpoint_path = sorted(
        dirs,
        key=lambda x: os.path.getmtime(x),
        reverse=True
    )[0]

    image_model.load_state_dict(torch.load(
        os.path.join(checkpoint_path, 'image_model.pth'),
        map_location=device
    ))

    text_model.load_state_dict(torch.load(
        os.path.join(checkpoint_path, 'language_model.pth'),
        map_location=device
    ))

    image_model.to(device).eval()
    text_model.to(device).eval()

    return args, image_model, text_model, dataset, loader, device


def save_metrics(metrics, results_dir, filename):
    metrics_dir = os.path.join(results_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    path = os.path.join(metrics_dir, filename)
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to {path}")

def evaluate_fashion_mnist_accuracy(image_model, text_model, dataset, tokenizer, device, results_dir):
    print("\nEvaluating Fashion MNIST Accuracy...")

    evaluator = MixedFashionMNISTEvaluator(
        image_model=image_model,
        text_model=text_model,
        dataset=dataset,
        tokenizer=tokenizer,
        device=device
    )

    image_to_text_acc, text_to_image_acc = evaluator.evaluate()

    print("\n--- Image-to-Text Accuracy ---")
    print(f"Top-1 Accuracy: {round(image_to_text_acc[0] * 100, 2)}%")
    print(f"Top-2 Accuracy: {round(image_to_text_acc[1] * 100, 2)}%")
    print(f"Top-3 Accuracy: {round(image_to_text_acc[2] * 100, 2)}%")

    print("\n--- Text-to-Image Accuracy ---")
    print(f"Top-1 Accuracy: {round(text_to_image_acc[0] * 100, 2)}%")
    print(f"Top-2 Accuracy: {round(text_to_image_acc[1] * 100, 2)}%")
    print(f"Top-3 Accuracy: {round(text_to_image_acc[2] * 100, 2)}%")

    os.makedirs(results_dir, exist_ok=True)
    metrics = {
        "image_to_text_accuracy_top1": image_to_text_acc[0],
        "image_to_text_accuracy_top2": image_to_text_acc[1],
        "image_to_text_accuracy_top3": image_to_text_acc[2],
        "text_to_image_accuracy_top1": text_to_image_acc[0],
        "text_to_image_accuracy_top2": text_to_image_acc[1],
        "text_to_image_accuracy_top3": text_to_image_acc[2],
    }

    with open(os.path.join(results_dir, "eval_fashion_mnist_accuracy.json"), "w") as f:
        json.dump(metrics, f, indent=4)

def evaluate_perplexity(image_model, text_model, loader, tokenizer, device, results_dir):
    print("\nEvaluating Mixed Perplexity...")
    evaluator = MixedPerplexityEvaluator(
        image_model, text_model, loader, tokenizer, device=device
    )

    image_to_text_perp, text_to_image_perp, original_text_perp = evaluator.evaluate()

    print(f"Image-to-text perplexity: {round(image_to_text_perp, 2)}")
    print(f"Text reconstruction perplexity: {round(text_to_image_perp, 2)}")
    print(f"Original text perplexity: {round(original_text_perp, 2)}")

    save_metrics({
        "image_to_text_perplexity": image_to_text_perp,
        "text_to_image_perplexity": text_to_image_perp,
        "original_text_perplexity": original_text_perp
    }, results_dir, "eval_mixed_perplexity.json")



def main():
    args_path = "args.json"
    args, image_model, text_model, dataset, loader, device = load_models_and_data(args_path)

    evaluate_fashion_mnist_accuracy(image_model, text_model, loader, dataset.tokenizer, device, args["results_dir"])

    evaluate_perplexity(image_model, text_model, loader, dataset.tokenizer, device, args["results_dir"])



if __name__ == "__main__":
    main()