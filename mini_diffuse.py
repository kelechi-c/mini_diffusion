import torchvision
import numpy as np
import torch
from torch.nn import functional as fnn
from torchvision import transforms
from datasets import load_dataset
from diffusers.utils.dummy_pt_objects import DDPMScheduler
from diffusers import schedulers.DDPMScheduler
from PIL import Image as pillow_image
from matplotlib import pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_data = load_dataset("tensorkelechi/sky_images")
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
    out_image = pillow_image.new("RGB", (size * len(images), size))
    for k, img in enumerate(img_list):
        out_image.paste(img.resize(size, size), (k * size, 0))
    return out_image


image_pp = transforms.Compose([
    transforms.Resize((image_size, image_size)),

])
