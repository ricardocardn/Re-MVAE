import os
import json
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from playground.readers.fashion_mnist_mixed_dataset.reader import Reader
from playground.architectures.convolutional_image_autoencoder_depth_3 import Builder as ImageBuilder
from playground.architectures.gru_seq2seq_bidirectional_enc import Builder as TextBuilder
from playground.evaluators import (
    MixedFIDEvaluator,
    ImageFIDEvaluator,
    MixedFashionMNISTEvaluator,
    MixedPerplexityEvaluator
)
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
        image_size=args["image_size"],
        input_channels=1,
        latent_dim=args["latent_dim"]
    )
    text_model = TextBuilder().build(
        vocab_size=len(tokenizer.vocab),
        embedding_dim=args["embedding_dim"],
        hidden_dim=args["hidden_dim"],
        latent_dim=args["latent_dim"],
        context_length=args["context_length"],
        num_layers=1
    )

    checkpoint_base = args["checkpoint_dir"]
    dirs = [os.path.join(checkpoint_base, d) for d in os.listdir(checkpoint_base)]
    dirs = [d for d in dirs if os.path.isdir(d)]

    checkpoint_path = sorted(
        dirs,
        key=lambda x: os.path.getmtime(x),
        reverse=True
    )[0]

    image_model.load_state_dict(torch.load(os.path.join(checkpoint_path, 'image_model.pth'), map_location=device))
    text_model.load_state_dict(torch.load(os.path.join(checkpoint_path, 'language_model.pth'), map_location=device))

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


def evaluate_mixed_fid(image_model, text_model, loader, device, results_dir):
    print("\nEvaluating Mixed FID...")
    evaluator = MixedFIDEvaluator(
        image_model=image_model,
        text_model=text_model,
        dataset=loader,
        image_size=(56, 56),
        device=device
    )
    fid_rec, fid_gen = evaluator.evaluate()
    print(f"FID Score for Reconstructed Images (Image Generation): {fid_rec:.2f}")
    print(f"FID Score for Generated Images (Image Generation): {fid_gen:.2f}")
    save_metrics({
        "FID_reconstructed": fid_rec,
        "FID_generated": fid_gen
    }, results_dir, "eval_mixed_fid.json")


def evaluate_image_fid(image_model, loader, device, results_dir):
    print("\nEvaluating Image FID...")
    evaluator = ImageFIDEvaluator(
        model=image_model,
        dataset=loader,
        device=device,
        image_size=(56, 56)
    )
    fid_rec, fid_gen = evaluator.evaluate()
    print(f"FID Score for Reconstructed Images (Image Reconstruction): {fid_rec:.2f}")
    print(f"FID Score for Generated Images (Image Reconstruction): {fid_gen:.2f}")
    save_metrics({
        "FID_reconstructed": fid_rec,
        "FID_generated": fid_gen
    }, results_dir, "eval_image_fid.json")


def evaluate_fashion_mnist_accuracy(image_model, text_model, loader, tokenizer, device, results_dir):
    print("\nEvaluating Mixed Fashion MNIST Accuracy...")
    evaluator = MixedFashionMNISTEvaluator(
        image_model, text_model, loader, tokenizer, device=device
    )
    image_to_text_acc, text_to_image_acc = evaluator.evaluate()
    print(f"Image-to-text accuracy: {round(image_to_text_acc * 100, 2)}%")
    print(f"Text-to-image accuracy: {round(text_to_image_acc * 100, 2)}%")
    save_metrics({
        "image_to_text_accuracy": image_to_text_acc,
        "text_to_image_accuracy": text_to_image_acc
    }, results_dir, "eval_fashion_mnist_accuracy.json")


def evaluate_perplexity(image_model, text_model, loader, tokenizer, device, results_dir):
    print("\nEvaluating Mixed Perplexity...")
    evaluator = MixedPerplexityEvaluator(
        image_model, text_model, loader, tokenizer, device=device
    )
    image_to_text_perp, text_to_image_perp = evaluator.evaluate()
    print(f"Image-to-text perplexity: {round(image_to_text_perp, 2)}")
    print(f"Text reconstruction perplexity: {round(text_to_image_perp, 2)}")
    save_metrics({
        "image_to_text_perplexity": image_to_text_perp,
        "text_to_image_perplexity": text_to_image_perp
    }, results_dir, "eval_mixed_perplexity.json")


def main():
    args_path = "args.json"
    args, image_model, text_model, dataset, loader, device = load_models_and_data(args_path)

    evaluate_mixed_fid(image_model, text_model, loader, device, args["results_dir"])
    evaluate_image_fid(image_model, loader, device, args["results_dir"])
    evaluate_fashion_mnist_accuracy(image_model, text_model, loader, dataset.tokenizer, device, args["results_dir"])
    evaluate_perplexity(image_model, text_model, loader, dataset.tokenizer, device, args["results_dir"])


if __name__ == "__main__":
    main()