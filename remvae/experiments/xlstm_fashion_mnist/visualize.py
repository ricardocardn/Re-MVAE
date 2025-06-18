import os
import sys
import json
import torch
import matplotlib.pyplot as plt
import textwrap
import seaborn as sns
from sklearn.manifold import TSNE
from functools import partial
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchvision import transforms

from readers.fashion_mnist_mixed_dataset.reader import Reader
from architectures.convolutional_image_autoencoder_depth_3 import Builder as ImageBuilder
from architectures.xlstm_seq2seq_bidirectional_enc_autoreg_enc import Builder as TextBuilder
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


def collate_fn(batch, pad_idx, max_len):
    images, sequences = zip(*batch)
    sequences = pad_sequence(
        [seq.detach().clone() if torch.is_tensor(seq) else torch.tensor(seq).detach().clone()
         for seq in sequences],
        batch_first=True,
        padding_value=pad_idx
    )
    if sequences.size(1) < max_len:
        pad_len = max_len - sequences.size(1)
        pad_tensor = torch.full((sequences.size(0), pad_len), fill_value=pad_idx, dtype=sequences.dtype)
        sequences = torch.cat([sequences, pad_tensor], dim=1)
    sequences = sequences[:, :max_len]
    images = torch.stack(images, dim=0)
    return images, sequences


def transform_to_text(output, tokenizer):
    return ' '.join(tokenizer.decode(output.numpy().tolist()[0]))


def reconstruct_text_from_image(image_model, text_model, loader, dataset, device, output_dir, image_size):
    fig, axs = plt.subplots(10, 2, figsize=(10, 40))
    count = 0

    for idx, (img, desc) in enumerate(loader):
        if count == 10:
            break

        img = img.to(device)
        desc = desc.to(device)

        with torch.no_grad():
            recon, mu, sigma = image_model(img)
            z = image_model.reparametrize(mu, sigma)
            text_recon = text_model.decode(z).argmax(dim=-1)

        recon_img = recon[0].reshape((1, image_size, image_size)).cpu()
        desc_text = transform_to_text(desc.cpu(), dataset.tokenizer).split(' <eos>')[0]
        recon_text = transform_to_text(text_recon.cpu(), dataset.tokenizer).split(' <eos>')[0]

        axs[idx, 0].imshow(img[0].permute(1, 2, 0).cpu().numpy())
        axs[idx, 1].imshow(recon_img.permute(1, 2, 0).numpy())

        axs[idx, 0].set_title("\n".join(textwrap.wrap(f"Original: {desc_text}", 40)), fontsize=10)
        axs[idx, 1].set_title("\n".join(textwrap.wrap(f"Reconstruido: {recon_text}", 40)), fontsize=10)

        axs[idx, 0].axis('off')
        axs[idx, 1].axis('off')

        count += 1

    plt.tight_layout(pad=4.0)
    path = os.path.join(output_dir, 'reconstruction_text_from_image.png')
    plt.savefig(path)
    plt.close()
    print(f"Image saved at: {path}")


def generate_image_from_text(image_model, text_model, loader, dataset, device, output_dir, image_size, images=10):
    for sample in range(images):
        count = 0
        fig, axs = plt.subplots(2, 5, figsize=(25, 20))
        axs = axs.flatten()

        for idx, (img, desc) in enumerate(loader):
            if count == 10:
                break

            desc = desc.to(device)

            with torch.no_grad():
                _, mu, sigma = text_model(desc)
                z = text_model.reparametrize(mu, sigma)
                img_recon = image_model.decode(z)

            b = img_recon[0].reshape((1, image_size, image_size)).cpu()
            desc_text = transform_to_text(desc.cpu(), dataset.tokenizer).split(' <eos>')[0]
            wrapped_text = "\n".join(textwrap.wrap(f"Prompt: {desc_text}", width=30))

            axs[idx].imshow(b.permute(1, 2, 0).numpy())
            axs[idx].set_title(wrapped_text, fontsize=20, pad=15)
            axs[idx].axis('off')

            count += 1

        plt.tight_layout(pad=4.0)
        path = os.path.join(output_dir, f'generated_image_from_text_{sample}.png')
        plt.savefig(path)
        plt.close()
        print(f"Image saved at: {path}")


def interpolate_images(image_model, loader, device, output_dir, image_size, rows=5, cols=10):
    fig, axs = plt.subplots(rows, cols, figsize=(30, 15))
    fig.subplots_adjust(hspace=0.4, wspace=0.2)
    axs = axs.flatten()

    loader_iter = iter(loader)

    for row in range(rows):
        try:
            img1, _ = next(loader_iter)
            img2, _ = next(loader_iter)
        except StopIteration:
            print("Not enough images in loader to interpolate.")
            break

        img1 = img1.to(device)
        img2 = img2.to(device)

        with torch.no_grad():
            _, mu1, sigma1 = image_model(img1)
            _, mu2, sigma2 = image_model(img2)

            z1 = image_model.reparametrize(mu1, sigma1)
            z2 = image_model.reparametrize(mu2, sigma2)

            alphas = torch.linspace(0, 1, cols)

            for col, alpha in enumerate(alphas):
                ax = axs[row * cols + col]

                if alpha.item() == 0:
                    img_to_show = img1[0]
                    title = "Original 1"
                elif alpha.item() == 1:
                    img_to_show = img2[0]
                    title = "Original 2"
                else:
                    z_interp = (1 - alpha) * z1 + alpha * z2
                    img_interp = image_model.decode(z_interp)
                    img_to_show = img_interp[0]
                    title = f"α={alpha:.2f}"

                img_to_show = img_to_show.reshape((1, image_size, image_size))
                ax.imshow(img_to_show.permute(1, 2, 0).detach().cpu().numpy())
                ax.axis('off')

                if row == 0:
                    ax.set_title(title, fontsize=12)

    plt.tight_layout()
    interp_path = os.path.join(output_dir, 'interpolation_grid.png')
    plt.savefig(interp_path)
    plt.close()
    print(f"Interpolation image saved at: {interp_path}")


def plot_tsne_latent_space(image_model, text_model, loader, dataset, device, output_dir):
    text_labels = [
        "t-shirt", "trouser", "pullover", "dress", "coat",
        "sandal", "shirt", "sneaker", "bag", "ankle boot"
    ]
    latents = []
    labels = []

    with torch.no_grad():
        for img, seq in loader:
            _, mu, _ = image_model(img.to(device))

            for i, s in enumerate(seq):
                text_decoded = transform_to_text(s.unsqueeze(0), dataset.tokenizer).lower()
                found_label = next((label for label in text_labels if label in text_decoded), None)

                if found_label:
                    latents.append(mu[i].cpu())
                    labels.append(found_label)

    if not latents:
        print("No labeled latents found for t-SNE.")
        return

    latents = torch.stack(latents, dim=0)

    text_embeddings = []
    with torch.no_grad():
        for label in text_labels:
            tokens = torch.tensor(dataset.tokenizer.encode(f"{label} <eos>")).unsqueeze(0).to(device)
            _, mu, _ = text_model(tokens)
            mu = mu.squeeze(0)
            text_embeddings.append(mu.cpu())

    text_embeddings = torch.stack(text_embeddings, dim=0)
    all_latents = torch.cat([latents, text_embeddings], dim=0)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    all_latents_2d = tsne.fit_transform(all_latents.numpy())

    img_latents_2d = all_latents_2d[:len(latents)]
    text_latents_2d = all_latents_2d[len(latents):]

    plt.figure(figsize=(15, 8))
    palette = sns.color_palette("tab10", len(text_labels))
    label_to_color = {label: palette[i] for i, label in enumerate(text_labels)}

    for label in text_labels:
        idxs = [i for i, l in enumerate(labels) if l == label]
        if idxs:
            coords = img_latents_2d[idxs]
            plt.scatter(coords[:, 0], coords[:, 1], label=label, alpha=0.8, s=20, color=label_to_color[label])

    for i, (x, y) in enumerate(text_latents_2d):
        plt.scatter(x, y, marker='X', s=200, color=palette[i], edgecolor='black')
        plt.text(x + 1.5, y, text_labels[i],
                 fontsize=14, fontweight='bold', color='black',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.3'))

    plt.title("Visualización del espacio latente (t-SNE):\nEmbeddings de texto e imágenes", fontsize=16)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(title="Etiqueta real", fontsize=14, loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.grid(True)

    path = os.path.join(output_dir, 'tsne_latent_space.png')
    plt.savefig(path)
    plt.close()
    print(f"t-SNE plot saved at: {path}")


def find_latest_checkpoint_dir(base_path):
    subdirs = [
        os.path.join(base_path, d) for d in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, d))
    ]
    if not subdirs:
        raise FileNotFoundError(f"Models not found in {base_path}")
    subdirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return subdirs[0]


def main():
    if len(sys.argv) != 2:
        print("Uso: python eval.py args.json")
        sys.exit(1)

    args_path = sys.argv[1]
    with open(args_path, 'r') as f:
        args = json.load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform = transforms.Compose([
        transforms.Resize((args["image_size"], args["image_size"])),
        transforms.ToTensor()
    ])

    dataset = Reader(train=False, transform=transform, len=args["dataset_length"])
    dataset.tokenizer = TextTokenizer.load(args["tokenizer_path"])
    pad_idx = dataset.tokenizer.token2idx['<pad>']

    loader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=partial(collate_fn, pad_idx=pad_idx, max_len=args["context_length"]),
        shuffle=False
    )

    image_model = ImageBuilder().build(
        args["image_size"],
        1,
        args["latent_dim"]
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


    checkpoint_base = args.get("checkpoint_dir")
    if checkpoint_base is None or not os.path.isdir(checkpoint_base):
        raise ValueError("Parameter 'checkpoint_dir' must be defined as a valid directory in args.json")

    checkpoint_dir = find_latest_checkpoint_dir(checkpoint_base)

    image_model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'image_model.pth'), map_location=device))
    text_model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'language_model.pth'), map_location=device))

    image_model.to(device).eval()
    text_model.to(device).eval()

    output_dir = os.path.join(os.path.dirname(args_path), 'images')
    os.makedirs(output_dir, exist_ok=True)

    reconstruct_text_from_image(image_model, text_model, loader, dataset, device, output_dir, args["image_size"])
    generate_image_from_text(image_model, text_model, loader, dataset, device, output_dir, args["image_size"])
    interpolate_images(image_model, loader, device, output_dir, args["image_size"])
    plot_tsne_latent_space(image_model, text_model, loader, dataset, device, output_dir)


if __name__ == "__main__":
    main()