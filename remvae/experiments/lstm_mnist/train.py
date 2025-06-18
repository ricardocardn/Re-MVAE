import json
import sys
import torch
import torch.nn as nn
from functools import partial
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchvision import transforms
from readers.mnist_mixed_dataset.reader import Reader
from architectures.convolutional_image_autoencoder_depth_3 import Builder as ImageBuilder
from architectures.lstm_seq2seq_bidirectional_enc import Builder as TextBuilder, Wrapper
from trainers import MixedAdaptativennealingTrainer


with open(sys.argv[1], 'r') as f:
    args = json.load(f)


transform = transforms.Compose([
    transforms.Resize((args["image_size"], args["image_size"])),
    transforms.ToTensor()
])
dataset = Reader(train=True, transform=transform, len=args["dataset_length"])
dataset.tokenizer.save(args["tokenizer_path"])
pad_idx = dataset.tokenizer.token2idx['<pad>']

def collate_fn(batch, max_len=args["context_length"]):
    images, sequences = zip(*batch)
    sequences = pad_sequence(
        [seq.detach().clone() if isinstance(seq, torch.Tensor) else torch.tensor(seq).detach().clone()
         for seq in sequences],
        batch_first=True, padding_value=pad_idx
    )
    if sequences.size(1) < max_len:
        pad_len = max_len - sequences.size(1)
        pad_tensor = torch.full((sequences.size(0), pad_len), fill_value=pad_idx, dtype=sequences.dtype)
        sequences = torch.cat([sequences, pad_tensor], dim=1)
    sequences = sequences[:, :max_len]
    images = torch.stack(images, dim=0)
    return images, sequences

loader = DataLoader(
    dataset,
    batch_size=args["batch_size"],
    collate_fn=partial(collate_fn, max_len=args["context_length"]),
    shuffle=True
)


image_model = ImageBuilder().build(
    args["image_size"], 1, args["latent_dim"]
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

params = list(image_model.parameters()) + list(text_model.parameters())
optimizer = torch.optim.Adam(params, lr=1e-3, betas=(0.5, 0.999), weight_decay=1e-5)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
image_model.to(device)
text_model.to(device)

trainer = MixedAdaptativennealingTrainer(
    wrapper,
    image_model,
    nn.CrossEntropyLoss(ignore_index=pad_idx, reduction='mean'),
    nn.BCELoss(reduction='mean'),
    optimizer,
    epochs=args["epochs"],
    latent_dim=args["latent_dim"],
    weights=args["weights"],
    method="modified",
    k=args["k"],
    x0=args["x0"]
)

_, _, metrics = trainer.train(
    loader,
    device,
    return_metrics=True,
    results_dir=args["results_dir"],
    checkpoint_dir=args["checkpoint_dir"],
    checkpoint_steps=args["checkpoint_steps"]
)