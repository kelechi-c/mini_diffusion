import torch
from torch import nn


class VAEBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.downsample = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1),
            nn.SiLU(),
            nn.GroupNorm(32, out_channels),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor):
        x = self.downsample(x)

        return x
