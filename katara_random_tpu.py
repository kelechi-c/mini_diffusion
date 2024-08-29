# minimal diffusion model for clouds, using huggingface diffusers architecture For my complete from-scratch implementation, check the the tiny_diffusion folder
# implemented in pytorch

import torchvision
import torch
import numpy as np
import cv2
from torch.nn import functional as fnn
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from diffusers import DDPMPipeline
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.models.unets import UNet2DModel
from PIL import Image as pillow_image
from tqdm.auto import tqdm
from einops import rearrange


# TPU specifics
import torch_xla as xla
import torch_xla.core.xla_model as xm

device = xla.device()

print(xla.devices())
print(device)


# general variables
huggan_data = "huggan/wikiart"
split = 1000
image_size = 128
batch_size = 32

paint_images = load_dataset(huggan_data, split="train", streaming=True)
paint_images = paint_images.take(split)


# utility functions
def read_image(img, img_size=128):
    img = np.array(img)
    img = cv2.resize(img, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0

    return img


def display_img(img):
    img = img * 0.5 + 0.5
    imgrid = torchvision.utils.make_grid(img)
    grid = imgrid.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255
    grid = pillow_image.fromarray(np.array(grid).astype(np.uint8))

    return grid


def image_grid(img_list: list, size=image_size):
    out_image = pillow_image.new("RGB", (size * len(img_list), size))
    for k, img in enumerate(img_list):
        out_image.paste(img.resize(size, size), (k * size, 0))
    return out_image


class ImageDataset(IterableDataset):
    def __init__(self, dataset=paint_images):
        super().__init__()
        self.dataset = dataset

    def __iter__(self):
        for item in self.dataset:
            image = read_image(item["image"])

            image = torch.tensor(image, dtype=torch.float32)

            yield image


img_dataset = ImageDataset()

train_dataloader = DataLoader(img_dataset, batch_size=batch_size)

x = next(iter(train_dataloader))
x.shape

# for sample display of images/shape check
sample = next(iter(train_dataloader))

sample.shape

# scheduler for noise addition
noise_scheduler = DDPMScheduler(
    num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2"
)

noise = torch.rand_like(sample)
timesteps = torch.linspace(0, 999, 32).long().to(device)
print(f"shape => noise {noise.shape}, timesteps - {timesteps.shape}")

noisy_sample = noise_scheduler.add_noise(sample, noise, timesteps).to(device)
print(f"shape of noised sample=> {noisy_sample.shape}")
print(f"shape of image sample=> {sample.shape}")

display_img(noisy_sample.permute(0, 3, 1, 2))

# define the unet model for downsampleing and upsampling
unet_model = UNet2DModel(
    sample_size=image_size,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(64, 128, 128, 256),
    down_block_types=(
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
    ),
    up_block_types=("AttnUpBlock2D", "AttnUpBlock2D",
                    "UpBlock2D", "UpBlock2D"),
)

unet_model = unet_model.to(device)

with torch.no_grad():
    x = rearrange(noisy_sample, "b h w c -> b c h w")
    pred = unet_model(x, timesteps).sample

print(pred.shape)

optimizer = torch.optim.AdamW(unet_model.parameters(), lr=4e-4)
losses = []
epochs = 10


# training loop
def _trainer():
    for epoch in tqdm(range(epochs)):
        print(f"training epoch @ {epoch + 1}")
        for step, image in tqdm(enumerate(train_dataloader)):
            with xla.step():
                image = rearrange(image, "b h w c -> b c h w")
                images = image.to(device)

                noise = torch.randn(images.shape).to(
                    device
                )  # noise to add to the images

                b = images.shape[0]

                timesteps = torch.randint(
                    0, noise_scheduler.num_train_timesteps, (b,), device=device
                ).long()  # random timestep
                noisy_images = noise_scheduler.add_noise(
                    images, noise, timesteps
                )  # add noise to images
                noise_pred = unet_model(noisy_images, timesteps, return_dict=False)[
                    0
                ]  # model prediction

                # loss function
                loss = fnn.mse_loss(noise_pred, noise)
                loss.backward()
                losses.append(loss.item())

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
        torch.save(checkpoint, f"mini_diffuse_check_{epoch}")

        print(f"epoch @ {epoch + 1} => loss: {epoch_loss}")
        print("_____________________________")


_trainer()

torch.save(unet_model.state_dict(), "mini_diffuse.pth")

# sample generation
rd_sample = torch.randn(5, 3, image_size, image_size).to(device)
print(rd_sample.shape)

for k, c in tqdm(enumerate(noise_scheduler.timesteps)):
    with torch.no_grad():
        residual = unet_model(rd_sample, c).sample  # model prediction

    rd_sample = noise_scheduler.step(residual, c, rd_sample).prev_sample

display_img(rd_sample)

# model pipeline for aving and upload
sky_diffuse_pipe = DDPMPipeline(
    unet=unet_model, scheduler=noise_scheduler
)  # .to(device)
sky_diffuse_pipe.save_pretrained("sky_diff")


print("training complete")
