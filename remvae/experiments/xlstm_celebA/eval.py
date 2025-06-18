import os
import json
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from readers.celeba_large_mixed_dataset.reader import Reader
from architectures.convolutional_norm_image_autoencoder import Builder as ImageBuilder
from architectures.xlstm_seq2seq_bidirectional_enc_autoreg_enc import Builder as TextBuilder
from evaluators import MixedFIDEvaluator, ImageFIDEvaluator
from utils import TextTokenizer

from omegaconf import OmegaConf
from dacite import from_dict, Config as DaciteConfig
from xlstm import xLSTMBlockStackConfig


def get_config(path, file):
    cfg = ''
    with open(path + file, 'r') as f:
        cfg += f.read()

    cfg = OmegaConf.create(cfg)
    return from_dict(data_class=xLSTMBlockStackConfig, data=OmegaConf.to_container(cfg), config=DaciteConfig(strict=True))


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

    path = 'experiments/xlstm_celebA/configs/'
    encoder_config = get_config(path, 'encoder.yml')
    decoder_config = get_config(path, 'decoder.yml')

    builder = TextBuilder()
    text_model = builder.build(
        vocab_size=len(dataset.tokenizer.vocab),
        latent_dim=args["latent_dim"],
        encoder_config=encoder_config,
        decoder_config=decoder_config
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