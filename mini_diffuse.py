# minimal diffusion model for clouds, using huggingface diffusers architecture rFor my complete from-scratch implementation, check the the tiny_diffusion folder
# implemented in pytorch
from time import time
import torchvision
import torch
from torch.nn import functional as fnn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import wandb
from PIL import Image as pillow_image
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange
from datasets import load_dataset
from diffusers.pipelines.ddpm.pipeline_ddpm import DDPMPipeline
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from huggingface_hub import (
    login,
    get_full_repo_name,
    HfApi,
    create_repo,
)


# config
class config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = 128
    batch_size = 32
    data_id = "Artificio/WikiArt"


paint_images = load_dataset("huggan/wikiart", split="train", streaming=True)
paint_images = paint_images.take(100)


image_pp = transforms.Compose(
    [
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


def image_transforms(data):
    image_data = [image_pp(image.convert("RGB")) for image in data["image"]]

    yield {"image": image_data}


# utility functions


def display_img(img):
    img = img * 0.5 + 0.5
    imgrid = torchvision.utils.make_grid(img)
    grid = imgrid.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255
    grid = pillow_image.fromarray(np.array(grid).astype(np.uint8))

    return grid


def image_grid(img_list: list, size=config.image_size):
    out_image = pillow_image.new("RGB", (size * len(img_list), size))
    for k, img in enumerate(img_list):
        out_image.paste(img.resize(size, size), (k * size, 0))

    return out_image


paint_images.map(image_transforms)
train_dataloader = DataLoader(paint_images, batch_size=config.batch_size)

# for sample display of images/shape check
sample = next(iter(train_dataloader))["image"].to(config.device)[:5]
display_img(sample.permute(0, 3, 1, 2))


# scheduler for noise addition
noise_scheduler = DDPMScheduler(
    num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2"
)

noise = torch.rand_like(sample)
timesteps = torch.linspace(0, 999, config.batch_size).long().to(config.device)
noisy_sample = noise_scheduler.add_noise(sample, noise, timesteps.long())

print(f"shape of noised sample=> {noisy_sample.shape}")
print(f"shape of image sample=> {sample.shape}")


# define the unet model for downsampleing and upsampling
unet_model = UNet2DModel(
    sample_size=config.image_size,
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

unet_model = unet_model.to(config.device)

with torch.no_grad():
    noisy_sample = rearrange(noisy_sample, "b h w c -> b c h w")
    pred = unet_model(noisy_sample, timesteps).sample

print(pred.shape)

# training loop
optimizer = torch.optim.AdamW(unet_model.parameters(), lr=4e-4)

losses = []
epochs = 30

# wandb init
wandb.login()
wandb.init(project="tiny_diffuse", name="minidiffuse_paint")
# lr_scheduler = get_cosine_schedule_with_warmup()

for epoch in tqdm(range(epochs)):
    print(f"training epoch @ {epoch + 1}")
    for step, image in tqdm(enumerate(train_dataloader)):
        image = rearrange(image, "b h w c -> b c h w")
        images = image.to(config.device)
        noise = torch.randn(images.shape).to(
            config.device
        )  # noise to add to the images
        b = images.shape[0]

        timesteps = torch.randint(
            0, noise_scheduler.num_train_timesteps, (b,), device=images.device
        ).long()  # random timestep
        noisy_images = noise_scheduler.add_noise(
            images, noise, timesteps.long()
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

    epoch_loss = sum(losses[-len(train_dataloader):]) / len(train_dataloader)
    print(f"epoch @ {epoch + 1} => loss: {epoch_loss}")

# plot the losses
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].plot(losses)
axs[1].plot(np.log(losses))
plt.show()

# save model as pt(better saved as pipeline)
try:
    torch.save(unet_model.state_dict(), "mini_diffuse2.pt")
except Exception as e:
    print(f"error saving model {e}")

# sample generation
rd_sample = torch.randn(5, 3, config.image_size,
                        config.image_size).to(config.device)
print(rd_sample.shape)

for k, c in tqdm(enumerate(noise_scheduler.timesteps)):
    with torch.no_grad():
        residual = unet_model(rd_sample, c).sample  # model prediction

    rd_sample = noise_scheduler.step(residual, c, rd_sample).prev_sample

display_img(rd_sample)  # show generated samples


login()

# model pipeline for aving and upload
sky_diffuse_pipe = DDPMPipeline(unet=unet_model, scheduler=noise_scheduler)
sky_diffuse_pipe.save_pretrained("sky_diff")
sky_diffuse_pipe.push_to_hub(
    "tensorkelechi/sky_diffuse")  # push pipeline directly
