import torch
from torch import nn
from torch.nn import functional as F

from runtime.vendor.attention.memory_encoder import _apply_rope, _rope_encoding


class TemporalAttention(nn.Module):
    def __init__(self, dim=384, heads=8):
        super().__init__()
        self.nh, self.hd = (heads, dim // heads)
        self.norm1 = nn.LayerNorm(dim)
        self.q, self.k, self.v = (nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim))
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.register_buffer("_frequency_cache", torch.empty(0), persistent=False)
        self._frequency_key = None

    def forward(self, sp, F_lat, n_special):
        b, length, dim = sp.shape
        h = self.norm1(sp)
        q = self.q(h).view(b, length, self.nh, self.hd).transpose(1, 2)
        k = self.k(h).view(b, length, self.nh, self.hd).transpose(1, 2)
        v = self.v(h).view(b, length, self.nh, self.hd).transpose(1, 2)
        key = (F_lat, n_special, sp.device, sp.dtype)
        if torch.compiler.is_compiling():
            positions = torch.arange(F_lat, device=sp.device).repeat_interleave(n_special)
            frequencies = _rope_encoding(self.hd, F_lat, sp.device, sp.dtype)[positions]
        elif self._frequency_key != key:
            positions = torch.arange(F_lat, device=sp.device).repeat_interleave(n_special)
            self._frequency_cache = _rope_encoding(self.hd, F_lat, sp.device, sp.dtype)[positions]
            self._frequency_key = key
            frequencies = self._frequency_cache
        else:
            frequencies = self._frequency_cache
        q, k = (_apply_rope(frequencies, q), _apply_rope(frequencies, k))
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        out = out.transpose(1, 2).reshape(b, length, dim)
        sp = sp + self.proj(out)
        return sp + self.mlp(self.norm2(sp))
