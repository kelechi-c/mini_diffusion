# UNet for the diffusion model

import torch
from torch import nn


class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
        )
