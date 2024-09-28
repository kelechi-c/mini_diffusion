# Dataloader and image utils
import torch
import torch_xla as xla
from skimage import io
import numpy as np
import cv2
import os
import shutil
import random
import requests
from datasets import Dataset, load_dataset
from torch.utils.data import IterableDataset, DataLoader


class config:
    batch_size = 8
    dtype = torch.float16
    img_size = 224
    latent_dim = 32
    attn_heads = 12
    learn_rate = 1e-4
    text_token_len = 77
    embed_dim = 768
    vocab_size = 49408
    quick_gelu_var = 1.702
    tpu_device = xla.device()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_image(img_url, img_size=512):
    img = io.imread(img_url) if isinstance(img_url, str) else np.array(img_url)
    img = cv2.resize(img, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0

    return img


# load LAION art dataset
dataset_id = "laion/laion-art"  # "CortexLM/midjourney-v6"
hfdata = load_dataset(dataset_id, split="train", streaming=True)
hfdata = hfdata.take(100_000)


class ImageDataset(IterableDataset):
    def __init__(self, dataset: Dataset = hfdata):
        super().__init__()
        self.dataset = dataset

    def __iter__(self):
        for item in self.dataset:
            image = read_image(item["URL"])
            caption = item["TEXT"]

            image = torch.tensor(image, dtype=config.dtype)
            caption = torch.tensor(caption, dtype=config.dtype)

            yield image, caption


dataset = ImageDataset()
train_loader = DataLoader(dataset, batch_size=config.batch_size)


def download(link, filename):  # fetch content from url
    file_res = requests.get(link, stream=True)
    image_file = os.path.basename(filename)

    with open(image_file, "wb") as file_writer:
        file_res.raw.decode_content = True
        # save to folder
        shutil.copyfileobj(file_res.raw, file_writer)

    return image_file


def seed_everything(seed=333):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
