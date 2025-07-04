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


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape."""
        in_dtype = x.dtype
        x = x.to(torch.float32)

        RMS = torch.sqrt(torch.sum(torch.square(x), dim=-1) / self.d_model + self.eps)

        result = einsum(x, self.g, "... d, d -> ... d") / torch.unsqueeze(RMS, dim=-1)

        return result.to(in_dtype)


def silu(x):
    return x * torch.sigmoid(x)


class Swiglu(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None):
        """composed of a SiLU activation function and a GLU"""
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 8 // 3
            d_ff = (d_ff // 64) * 64
        sigma = (2 / (d_model + d_ff)) ** 0.5
        self.w1 = nn.Parameter(nn.init.trunc_normal_(torch.randn((d_ff, d_model)) * sigma, a=-3 * sigma, b=3 * sigma))
        self.w2 = nn.Parameter(nn.init.trunc_normal_(torch.randn((d_model, d_ff)) * sigma, a=-3 * sigma, b=3 * sigma))
        self.w3 = nn.Parameter(nn.init.trunc_normal_(torch.randn((d_ff, d_model)) * sigma, a=-3 * sigma, b=3 * sigma))

    def forward(self, x):
        W1x = einsum(self.w1, x, "d_ff d_model, ... d_model -> ... d_ff")
        W3x = einsum(self.w3, x, "d_ff d_model, ... d_model -> ... d_ff")
        print(x.shape, self.w1.shape, W1x.shape)
        w1xsilu = silu(W1x)

        # to fix: a mess and slow
        return einsum(
            self.w2, (einsum(w1xsilu, W3x, "... d_ff, ... d_ff -> ... d_ff")), "d_model d_ff, ... d_ff -> ... d_model"
        )
