# minimal diffusion model for clouds, using huggingface diffusers. For my complete from-scratch implementation, check the the tiny_diffusion folder
# implemented in pytorch
import torchvision
import numpy as np
import torch
from torch.nn import functional as fnn
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.models.unets import UNet2DModel
from PIL import Image as pillow_image
from matplotlib import pyplot as plt

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
