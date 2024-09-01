import torch
from torch import nn
from .attention import VaeAttention


class VaeBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(32, out_channels, kernel_size=5, stride=2),
        )

        self.residual_layer = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        )

    def forward(self, x: torch.Tensor):
        res = x

        x = self.downsample(x)

        return x + self.residual_layer(res)


class VaeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_layers = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, stride=1),
            VaeBlock(128, 128),
            VaeBlock(128, 128),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            VaeBlock(128, 256),
            VaeBlock(256, 256),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=0),
            VaeBlock(512, 512),
            VaeBlock(512, 512),
            VaeAttention(512),  # attention section in VAE encoder
            VaeBlock(512, 512),
            nn.GroupNorm(32, 512),  # group normalization
            nn.SiLU(),  # swish activation btw
            nn.Conv2d(512, 8, kernel_size=3, padding=1),  # 512 output channels
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
        )

    def forward(self, x, noise):
        x = self.encoder_layers(x)
        mean, log_var = torch.chunk(x, 2, dim=1)  # split into two sections
        log_var = torch.clamp(-30, 20)
        var = log_var.exp()
        stdev = var.sqrt()

        x = mean + stdev * noise

        return x * 0.18215  # variable defined in the repo


class VaeDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder_layers = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            nn.Conv2d(4, 512, kernel_size=3, padding=1),  # height / 8
            VaeBlock(512, 512),
            VaeBlock(512, 512),
            VaeBlock(512, 512),
            VaeAttention(512),
            VaeBlock(512, 512),
            nn.Upsample(scale_factor=2),  # height / 4
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VaeBlock(512, 512),
            VaeBlock(512, 512),
            nn.Upsample(scale_factor=2),  # height / 2
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VaeBlock(512, 256),
            VaeBlock(256, 256),
            VaeBlock(256, 256),
            nn.Upsample(scale_factor=2),  # original height/resolution
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            VaeBlock(256, 128),
            VaeBlock(128, 128),
            VaeBlock(128, 128),
            nn.GroupNorm(32, 128),
            nn.SiLU(),  # swish, sigmpoid linear unit
            nn.Conv2d(128, 3, kernel_size=3, stride=2, padding=0),
        )

    def forward(self, x: torch.Tensor):
        x = x / 0.18215  # divide to return to normal
        x = self.decoder_layers(x)

        return x


class Vae(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VaeEncoder()
        self.decoder = VaeDecoder()

    def encoder(self, image: torch.Tensor) -> torch.Tensor:
        return self.encoder(image)

    def decoder(self, image: torch.Tensor) -> torch.Tensor:
        return self.decoder(image)
