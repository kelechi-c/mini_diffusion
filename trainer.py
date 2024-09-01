# script for model training, checkpointing

import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from einops import rearrange
from torch.nn import functional as fnn
from tqdm.auto import tqdm
from katara.data_utils import config

optimizer = optim.AdamW()

checkpoint = torch.load(path, weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
epoch = checkpoint["epoch"]
loss = checkpoint["loss"]


# training loop
def _trainer(model: nn.Module, trainloader: DataLoader, epochs):
    losses = []
    for epoch in tqdm(range(epochs)):
        print(f"training epoch @ {epoch + 1}")
        for _, image in tqdm(enumerate(train_dataloader)):
            image = rearrange(image, "b h w c -> b c h w")
            images = image.to(device)

            # loss function
            loss = fnn.mse_loss(noise_pred, noise)
            loss.backward()

            # gradient update
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss = sum(losses[-split:]) / split
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": unet_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_loss,
        }
        torch.save(checkpoint, f"katara_mini_check_{epoch}.pth")

        print(f"epoch @ {epoch + 1} => loss: {epoch_loss}")
        print("_____________________________")
