import torch
from torch import nn
from torch.nn import functional as func_nn


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


class TImeEmbedding(nn.Module):
    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.linear_1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.linear_2 = nn.Linear(4 * embed_dim, embed_dim)

    def forward(self, x: torch.Tensor):
        x = self.linear_1(x)  # 4 x embed_dim

        x = func_nn.silu(x)  # swish activation
        x = self.linear_2(x)  # rescale to normal embed dim

        return x
