import os
import json
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from playground.readers.CelebAMixedLargeDataset.reader import Reader
from playground.architectures.ConvolutionalNormImageAutoencoder import Builder as ImageBuilder
from playground.architectures.xLSTMSeq2seqBidirectionalAutoregressive import Builder as TextBuilder, Wrapper
from playground.evaluators import (MixedFIDEvaluator, MixedPerplexityEvaluator)
from playground.helpers.tokenizer import TextTokenizer

from omegaconf import OmegaConf
from dacite import from_dict, Config as DaciteConfig
from xlstm import xLSTMBlockStackConfig


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
        args["image_size"], args["input_channels"], args["latent_dim"], args["conv_dims"]
    )

    path = args["config_path"]
    encoder_config = from_dict(data_class=xLSTMBlockStackConfig, data=OmegaConf.to_container(OmegaConf.create(open(path + '/encoder.yml', 'r').read())), config=DaciteConfig(strict=True))
    decoder_config = from_dict(data_class=xLSTMBlockStackConfig, data=OmegaConf.to_container(OmegaConf.create(open(path + '/decoder.yml', 'r').read())), config=DaciteConfig(strict=True))
    
    builder = TextBuilder()
    text_model = builder.build(
        vocab_size=len(dataset.tokenizer.vocab),
        latent_dim=args["latent_dim"],
        encoder_config=encoder_config,
        decoder_config=decoder_config
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

    evaluate_mixed_fid(image_model, text_model, loader, device, args["results_dir"])

    evaluate_perplexity(image_model, text_model, loader, dataset.tokenizer, device, args["results_dir"])



if __name__ == "__main__":
    main()