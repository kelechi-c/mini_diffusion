# minimal diffusion model for clouds, using huggingface diffusers. For my complete from-scratch implementation, check the the tiny_diffusion folder
# implemented in pytorch

from time import time
import torchvision
import torch
from torch.nn import functional as fnn
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import load_dataset
from diffusers import DDPMPipeline
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.models.unets import UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from huggingface_hub import (
    notebook_login,
    get_full_repo_name,
    HfApi,
    create_repo,
    ModelCard,
)

import numpy as np
import wandb
from PIL import Image as pillow_image
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

# geeral variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sky_image_data = load_dataset("tensorkelechi/sky_images")
image_size = 64
batch_size = 32

# utility functions


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


image_pp = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


def image_transforms(data):
    image_data = [image_pp(image.convert("RGB")) for image in data["image"]]

    return {"image": image_data}


sky_image_data.set_transform(image_transforms)
train_dataloader = DataLoader(sky_image_data, batch_size=batch_size, shuffle=True)

# for sample display of images/shape check
sample = next(iter(train_dataloader))["image"].to(device)[:8]
display_img(sample).resize((8 * image_size, image_size), resample=pillow_image.NEAREST)

# scheduler for noise addition
noise_scheduler = DDPMScheduler(
    num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2"
)

noise = torch.rand_like(sample)
timesteps = torch.linspace(0, 999, 8).long().to(device)
noisy_sample = noise_scheduler.add_noise(sample, noise, timesteps)
print(f"shape of noised sample=> {noisy_sample.shape}")
print(f"shape of image sample=> {sample.shape}")

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
    up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
)

unet_model.to(device)

with torch.no_grad():
    pred = unet_model(noisy_sample, timesteps).sample

print(pred.shape)

# training loop
optimizer = torch.optim.AdamW(unet_model.parameters(), lr=4e-4)

losses = []
epochs = 30
wandb.login()
wandb.init(project="tiny_diffuse", name="minidiffuse_sky1_64x64")
# lr_scheduler = get_cosine_schedule_with_warmup()

for epoch in tqdm(range(epochs)):
    print(f"training epoch @ {epoch + 1}")
    for step, batch in tqdm(enumerate(train_dataloader)):
        images = batch["image"].to(device)
        noise = torch.randn(images.shape).to(device)  # noise to add to the images
        b = images.shape[0]

        timesteps = torch.randint(
            0, noise_scheduler.num_train_timesteps, (b,), device=images.device
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
        wandb.log({"loss": loss})

        # gradient update
        optimizer.step()
        optimizer.zero_grad()

    epoch_loss = sum(losses[-len(train_dataloader) :]) / len(train_dataloader)
    print(f"epoch @ {epoch + 1} => loss: {epoch_loss}")

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].plot(losses)
axs[1].plot(np.log(losses))
plt.show()

try:
    torch.save(unet_model.state_dict(), "mini_diffuse.pt")
except Exception as e:
    print(f"error saving model {e}")

# sample generation
rd_sample = torch.randn(10, 3, image_size, image_size)

for k, c in tqdm(enumerate(noise_scheduler.timesteps)):
    with torch.no_grad():
        residual = unet_model(sample, c).sample  # model prediction

    sample = noise_scheduler.step(residual, c, sample).prev_sample

display_img(sample)

# model pipeline for aving and upload
sky_diffuse_pipe = DDPMPipeline(unet=unet_model, scheduler=noise_scheduler)
sky_diffuse_pipe.save_pretrained("sky_diff")


# push model to hub
model_name = "sky_diffuse"
model_id = get_full_repo_name(model_name)

notebook_login()
create_repo(model_id)


hf_api = HfApi()
hf_api.upload_folder(
    folder_path="sky_diff/scheduler", path_in_repo="", repo_id=model_id
)
hf_api.upload_folder(folder_path="sky_diff/unet", path_in_repo="", repo_id=model_id)
hf_api.upload_file(
    path_or_fileobj="sky_diff/model_index.json",
    path_in_repo="model_index.json",
    repo_id=model_id,
)


# readme

content = f"""
---
license: apache 2.0
tags:
- pytorch
- diffusers
- unconditional-image-generation
- diffusion-models-class
---

This model is a diffusion model for unconditional image generation of clouds, skies, etc
## Usage```python
from diffusers import DDPMPipeline

pipeline = DDPMPipeline.from_pretrained('{model_id}')
image = pipeline().images[0]
image
"""


card = ModelCard(content)
card.push_to_hub(model_id)
