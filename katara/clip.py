import torch
from torch import nn
from .embeddings import ClipEmbeddings
from .attention import SelfAttention


class ClipLayer(nn.Module):
    def __init__(self, embed_dim: int = 768, attn_heads: int = 12):
        super().__init__()
        self.layernorm = nn.LayerNorm(embed_dim)
        self.attention = SelfAttention(attn_heads, embed_dim)
        self.linear_1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.linear_2 = nn.Linear(4 * embed_dim, embed_dim)

    def forward(self, x: torch.Tensor):
        residue_a = x
        x = self.layernorm(x)  # layernorm 1
        # self attention with causal mask
        x = self.attention(x, causal_mask=True)
        x = x + residue_a  # first residual connection

        # feed forward layer
        residue_b = x

        x = self.layernorm(x)
        x = self.linear_1(x)  # map to higher dim space
        x = x * torch.sigmoid(1.702 * x)  # QuickGELU activation
        x = self.linear_2(x)
        x = x + residue_b  # second residual connection

        return x


class ClipEncoder(nn.Module):
    def __init__(self, n_layers: int = 12, embed_dim: int = 768):
        super().__init__()
        self.embedding = ClipEmbeddings()
        self.norm_layer = nn.LayerNorm(embed_dim)

        # clip-text encoding layers
        self.layers = [ClipLayer() for _ in range(n_layers)]
        self.clip_layers = nn.Sequential(*self.layers)

    def forward(self, x_token: torch.LongTensor) -> torch.Tensor:
        x_embed = self.embedding(x_token)
        x = self.clip_layers(x_embed)
        x = self.norm_layer(x)

        return x
