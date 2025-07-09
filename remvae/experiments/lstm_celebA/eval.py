import os
import json
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from playground.readers.CelebAMixedLargeDataset.reader import Reader
from playground.architectures.ConvolutionalNormImageAutoencoder import Builder as ImageBuilder
from playground.architectures.LSTMSeq2seqBidirectional import Builder as TextBuilder
from playground.evaluators import MixedFIDEvaluator, ImageFIDEvaluator
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
        input_channels=3,
        latent_dim=args["latent_dim"],
        conv_dims=args.get("conv_dims", None)
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
    checkpoint_path = sorted(
        [os.path.join(checkpoint_base, d) for d in os.listdir(checkpoint_base)],
        key=lambda x: os.path.getmtime(x),
        reverse=True
    )[0]

    image_model.load_state_dict(torch.load(os.path.join(checkpoint_path, 'image_model.pth'), map_location=device))
    text_model.load_state_dict(torch.load(os.path.join(checkpoint_path, 'language_model.pth'), map_location=device))

    image_model.to(device).eval()
    text_model.to(device).eval()

    return image_model, text_model, dataset, loader, device


def evaluate_mixed_fid(image_model, text_model, loader, device):
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


def evaluate_image_fid(image_model, loader, device):
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


def main():
    args_path = "args.json"
    image_model, text_model, dataset, loader, device = load_models_and_data(args_path)

    evaluate_mixed_fid(image_model, text_model, loader, device)
    evaluate_image_fid(image_model, loader, device)


if __name__ == "__main__":
    main()