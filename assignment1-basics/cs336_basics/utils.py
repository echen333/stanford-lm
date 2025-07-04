import torch
import torch.nn as nn
from einops import einsum, rearrange


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """Construct a linear transformation module"""
        super().__init__()
        sigma = 2 / (in_features + out_features) ** 0.5
        self.weight = nn.Parameter(
            nn.init.trunc_normal_(
                torch.randn((out_features, in_features), dtype=dtype, device=device) * sigma, a=-3 * sigma, b=3 * sigma
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.weight, x, "out in, ... in -> ... out")


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
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape."""
        in_dtype = x.dtype
        x = x.to(torch.float32)

        RMS = torch.sqrt(torch.sum(torch.square(x), dim=-1) / self.d_model + self.eps)

        result = einsum(x, self.weight, "... d, d -> ... d") / torch.unsqueeze(RMS, dim=-1)

        return result.to(in_dtype)


def silu(x):
    return x * torch.sigmoid(x)


def softmax(x, dim):
    mx = torch.max(x, dim, keepdim=True)[0]
    tmp_x = x - mx
    exp_x = torch.exp(tmp_x)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)


class Swiglu(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None):
        """composed of a SiLU activation function and a GLU"""
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 8 // 3
            d_ff = (d_ff // 64) * 64
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x):
        W1x = self.w1(x)
        W3x = self.w3(x)
        w1xsilu = silu(W1x)

        return self.w2(einsum(w1xsilu, W3x, "... d_ff, ... d_ff -> ... d_ff"))


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        assert d_k % 2 == 0
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # If you would like to optimize it, you may use a
        # single RoPE module referenced by all layers, and it can have a 2d pre-computed buffer of sin and cos values
        # created during init with self.register_buffer(persistent=False), instead of a nn.Parameter (because
        # we do not want to learn these fixed cosine and sine values).

        pass

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape.

        Note that you should tolerate x with an arbitrary number of batch dimensions. You should assume that the token positions are a tensor of shape (..., seq_len) specifying the token positions of x along the sequence dimension. You should use the token positions to slice your (possibly precomputed) cos and sin tensors along the sequence dimension."""

        k = self.d_k
        denom = 1.0 / (self.theta ** (torch.arange(0, k, 2).float() / k))
        thetas = einsum(token_positions.float(), denom, "... s, k2 -> ... s k2")

        cos_t = torch.cos(thetas)
        sin_t = torch.sin(thetas)

        rot_T = torch.stack((cos_t, sin_t, -sin_t, cos_t), dim=-1)
        from einops import rearrange

        tmp_x = rearrange(x, "... s (k2 r1) -> ... s k2 r1", r1=2)
        rot_T = rearrange(rot_T, "... s k2 (r1 r2) -> ... s k2 r1 r2", r1=2, r2=2)

        ret1 = einsum(tmp_x, rot_T, "...  s k2 r1, ... s k2 r1 r2 -> ... s k2 r2")
        print(ret1.shape)
        ret1 = ret1.flatten(start_dim=-2)
        return ret1


def scaled_dot_product(Q, K, V, mask=None):
    d = Q.shape[-1]
    dot_prod = einsum(Q, K, "b ... s1 d, b ... s2 d -> b ... s1 s2") * (d**-0.5)
    if mask is not None:
        dot_prod = dot_prod.masked_fill(mask == 0, -float("inf"))

    ret = einsum(softmax(dot_prod, -1), V, "b ... s1 s2, b ... s2 d -> b ... s1 d")
    return ret


class MultiHead_Self_Attention(nn.Module):
    def __init__(self, d_model, num_heads, theta=None, max_seq_len=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)

        self.rope = None
        if theta is not None and max_seq_len is not None:
            self.rope = RoPE(theta, self.d_k, max_seq_len=None)

    def forward(self, x, token_positions=None):
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = rearrange(Q, "... seq (h dk) -> h ... seq dk", h=self.num_heads)
        K = rearrange(K, "... seq (h dk) -> h ... seq dk", h=self.num_heads)
        V = rearrange(V, "... seq (h dk) -> h ... seq dk", h=self.num_heads)

        if self.rope is not None and token_positions is not None:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        seq = Q.shape[-2]
        mask = ~torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)

        ret = scaled_dot_product(Q, K, V, mask=mask)
        ret = rearrange(ret, "h ... seq dk -> ... seq (h dk)")  # ??
        ret = self.output_proj(ret)

        return ret


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, theta=None, max_seq_len=None):
        super().__init__()

        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.attn = MultiHead_Self_Attention(d_model, num_heads, theta, max_seq_len)
        self.ffn = Swiglu(d_model, d_ff)

    def forward(self, x):
        rms = self.ln1(x)
        y = x + self.attn(rms)
        rms2 = self.ln2(y)
        y2 = self.ffn(rms2)

        return y + y2
