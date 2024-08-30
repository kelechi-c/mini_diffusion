# Dataloader and image utils
import torch
from skimage import io
import numpy as np
import cv2
import os
import shutil
import requests
from datasets import load_dataset
from torch.utils.data import IterableDataset


class config:
    batch_size = 32
    dtype = torch.float16
    img_size = 224
    latent_dim = 32
    text_token_len = 77


def read_image(img_url, img_size=512):
    img = io.imread(img_url) if isinstance(str, img_url) else np.array(img_url)
    img = cv2.resize(img, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0

    return img


dataset_id = "CortexLM/midjourney-v6"
hfdata = load_dataset(dataset_id, split="train", streaming=True)
hfdata = hfdata.take(100000)


class ImageDataset(IterableDataset):
    def __init__(self, dataset=hfdata):
        super().__init__()
        self.dataset = dataset

    def __iter__(self):
        for item in self.dataset:
            image = read_image(item["image_url"])
            caption = item["prompt"]

            image = torch.tensor(image, dtype=config.dtype)
            caption = torch.tensor(caption, dtype=config.dtype)

            yield image, caption


def download(link, filename):  # fetch content from url
    file_res = requests.get(link, stream=True)
    image_file = os.path.basename(filename)

    with open(image_file, "wb") as file_writer:
        file_res.raw.decode_content = True
        # save to folder
        shutil.copyfileobj(file_res.raw, file_writer)

    return image_file