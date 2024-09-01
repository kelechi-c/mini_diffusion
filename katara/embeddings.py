import torch
from torch import nn


class ClipEmbeddings(nn.Module):
    def __init__(
        self, n_tokens: int = 77, embed_dim: int = 768, vocab_size: int = 49408
    ):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(vocab_size, embed_dim))
        self.token_embedding = nn.Embedding(n_tokens, embed_dim)

    def forward(self, x: torch.Tensor):
        x = self.token_embedding(x)

        return x + self.pos_embed
