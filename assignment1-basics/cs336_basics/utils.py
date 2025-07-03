import torch
import torch.nn as nn
from einops import einsum


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """Construct a linear transformation module"""
        super().__init__()
        sigma = 2 / (in_features + out_features) ** 0.5
        self.W = nn.Parameter(
            nn.init.trunc_normal_(
                torch.randn((out_features, in_features), dtype=dtype, device=device) * sigma, a=-3 * sigma, b=3 * sigma
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.W, x, "out in, ... in -> ... out")


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.emb = nn.Parameter(
            nn.init.trunc_normal_(torch.randn(num_embeddings, embedding_dim, device=device, dtype=dtype), a=-3, b=3)
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.emb[token_ids]
