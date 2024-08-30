import torch
import math
from torch import nn
from torch.nn import functional as func_nn
from einops import rearrange


class SelfAttention(nn.Module):
    def __init__(self, n_heads, embed_dim):
        self.n_heads = n_heads  # self-attention attention heads
        self.head_dim = embed_dim // n_heads  # split the dimension among the heads

        # input linear projection
        self.in_project = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.out_project = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x: torch.Tensor, causal_mask=True):
        batch_size, seq_len, embed_dim = x.shape

        x = self.in_project(x)

        # split query, key and value matrices from input tensor
        q, k, v = x.chunk(3, dim=-1)

        # reshape/unfold tensors
        q = rearrange(q, "b l (h d) -> b h l d")
        k = rearrange(k, "b l (h d) -> b h l d")
        v = rearrange(v, "b l (h d) -> b h l d")

        attn_weight = q @ k.transpose(-1, -2)  # attention matmul

        if causal_mask:  # apply causal_mask if satisfied
            attn_mask = torch.ones_like(
                attn_weight, dtype=torch.bool
            )  # mask of scalar 1's
            attn_mask = attn_mask.triu(1)  # triangular mask
            # apply mask to attention weights
            attn_weight.masked_fill_(attn_mask, -torch.inf)

        # divide by head dim, d from the attention equation
        attn_weight /= math.sqrt(self.head_dim)
        attn_score = func_nn.softmax(attn_weight, dim=1)

        output = attn_score @ v  # multiply with value matrix
        # shape -> (batch_size, sequence_length, dimension)
        output = rearrange(output, "b h l d -> b l (h d)")
        output = self.out_project(output)  # output linear projection

        return output


class CrossAttention(nn.Module):
    def __init__(self, n_heads):
        self.n_heads = n_heads

    def forward(self, x: torch.Tensor):
        return x
