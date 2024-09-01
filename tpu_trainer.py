from torch.utils.data import DataLoader
import torch_xla as xla
from torch import nn, optim
import torch_xla.core.xla_model as xm
from einops import rearrange
import torch
from torch.nn import functional as fnn
from tqdm.auto import tqdm
from einops import rearrange


# training loop
def _trainer(model: nn.Module, trainloader: DataLoader, epochs):
    for epoch in tqdm(range(epochs)):
        print(f"training epoch @ {epoch + 1}")
        for _, image in tqdm(enumerate(train_dataloader)):
            with xla.step():
                image = rearrange(image, "b h w c -> b c h w")
                images = image.to(device)

                # loss function
                loss = fnn.mse_loss(noise_pred, noise)
                loss.backward()

                # gradient update
                xm.optimizer_step(optimizer)
                xm.mark_step()
                optimizer.zero_grad()

        epoch_loss = sum(losses[-split:]) / split
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": unet_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_loss,
        }
        torch.save(checkpoint, f"katara_mini_check_{epoch}")

        print(f"epoch @ {epoch + 1} => loss: {epoch_loss}")
        print("_____________________________")
