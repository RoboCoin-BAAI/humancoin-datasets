import torch
from torch import nn

from .sdpa import attention


def sinusoidal_embedding_1d(dim, position):
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@torch.amp.autocast("cuda", enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


def rope_apply_qk(q, k, freqs):

    def apply(values):
        complex_values = torch.view_as_complex(values.float().reshape(*values.shape[:-1], -1, 2))
        return torch.view_as_real(complex_values * freqs).flatten(-2).to(values.dtype)

    return (apply(q), apply(k))


class WanRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-05):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim, eps=1e-06, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        return self._norm(x).type_as(x)

    def _norm(self, x):
        if self.weight is not None:
            x = super().forward(x.to(self.weight.dtype))
            return x
        else:
            return super().forward(x)


class WanSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-06):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, freqs, dtype=torch.bfloat16):
        b, s, n, d = (*x.shape[:2], self.num_heads, self.head_dim)

        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return (q, k, v)

        q, k, v = qkv_fn(x)
        q, k = rope_apply_qk(q, k, freqs)
        x = attention(q.to(dtype), k.to(dtype), v.to(dtype))
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanCrossAttention(WanSelfAttention):
    def forward(self, x, context, dtype=torch.bfloat16):
        b, n, d = (x.size(0), self.num_heads, self.head_dim)
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        x = attention(q.to(dtype), k.to(dtype), v.to(dtype))
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-06,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = (
            WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        )
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim)
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, x, e, freqs, context, dtype=torch.bfloat16):
        if e.dim() > 3:
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
            e = [e.squeeze(2) for e in e]
        else:
            e = (self.modulation + e).chunk(6, dim=1)
        temp_x = self.norm1(x) * (1 + e[1]) + e[0]
        temp_x = temp_x.to(dtype)
        y = self.self_attn(temp_x, freqs, dtype)
        x = x + y * e[2]

        def cross_attn_ffn(x, context, e):
            x = x + self.cross_attn(self.norm3(x).to(x.dtype), context, dtype)
            temp_x = self.norm2(x) * (1 + e[4]) + e[3]
            temp_x = temp_x.to(dtype)
            y = self.ffn(temp_x)
            x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, context, e)
        return x


class FeatureTransformer(nn.Module):
    def __init__(
        self,
        dim=3072,
        ffn_dim=14336,
        freq_dim=256,
        in_dim=148,
        num_heads=24,
        num_layers=16,
        text_len=512,
        eps=1e-06,
    ):
        super().__init__()
        self.dim, self.freq_dim, self.text_len = (dim, freq_dim, text_len)
        self.patch_embedding = nn.Conv3d(in_dim, dim, (1, 2, 2), stride=(1, 2, 2))
        self.text_embedding = nn.Sequential(
            nn.Linear(4096, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList(
            [
                WanAttentionBlock(dim, ffn_dim, num_heads, (-1, -1), True, True, eps)
                for _ in range(num_layers)
            ]
        )
        d = dim // num_heads
        self.freqs = torch.cat(
            (
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
            ),
            dim=1,
        )
