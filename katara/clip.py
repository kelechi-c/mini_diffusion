import torch
from torch import nn
from .embeddings import ClipEmbeddings


class ClipLayer(nn.Module):
    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.layernorm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        x = self.layernorm(x)

        return x


class ClipEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = ClipEmbeddings()
        self.layers = ClipLayer()

    def forward(self, x: torch.Tensor):
        x = self.token_embedding(x)

        return x + self.pos_embed
