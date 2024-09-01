import torch
import math
from torch import nn
from torch.nn import functional as func_nn
from einops import rearrange


class SelfAttention(nn.Module):
    def __init__(self, n_heads, embed_dim):
        super().__init__()
        self.n_heads = n_heads  # self-attention attention heads
        self.head_dim = embed_dim // n_heads  # split the dimension among the heads

        # input linear projection
        self.in_project = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.out_project = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x: torch.Tensor, causal_mask=True):
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

        output = attn_score @ v  # multiply with value matrixA

        # reshape -> (batch_size, sequence_length, dimension)
        output = rearrange(output, "b h l d -> b l (h d)")

        output = self.out_project(output)  # output linear projection

        return output


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, cross_dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads  # split the dimension among the heads

        # input linear projection
        self.q_project = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_project = nn.Linear(cross_dim, embed_dim, bias=True)
        self.v_project = nn.Linear(cross_dim, embed_dim, bias=True)

    def forward(self, x_input: torch.Tensor, y_target: torch.Tensor):
        q = self.q_project(x_input)  # q -> latent
        k = self.k_project(y_target)  # key -> context, text, mapping
        v = self.v_project(y_target)  # context value matrices

        # reshape/unfold tensors
        q = rearrange(q, "b l (h d) -> b h l d")
        k = rearrange(k, "b l (h d) -> b h l d")
        v = rearrange(v, "b l (h d) -> b h l d")

        attn_weight = q @ k.transpose(-1, -2)  # multiply query / key values

        # divide by head dim, d from the attention equation
        attn_weight /= math.sqrt(self.head_dim)
        attn_score = func_nn.softmax(attn_weight, dim=1)  # apply spftmax

        output = attn_score @ v  # multiply with value matrix

        # shape -> (batch_size, sequence_length, dimension)
        # reshape for output
        output = rearrange(output, "b h l d -> b l (h d)")
        output = self.out_project(output)  # output projection

        return output


class VaeAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor):
        residue = x
        bs, ch, h, w = x.shape  # batch_size, channels, height, width
        x = self.groupnorm(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.attention(x)  # apply self-attention without causal mask

        x = rearrange(x, "b (h w) c -> b c h w")

        return x + residue
