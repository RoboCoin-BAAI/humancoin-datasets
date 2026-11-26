from __future__ import annotations

import torch
from torch import nn


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).reshape(*x.shape[:-2], -1)


def _rope_encoding(head_dim: int, n_pos: int, device, dtype) -> torch.Tensor:
    t = torch.arange(n_pos, device=device, dtype=torch.float32)
    freqs = 1.0 / 10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    f = torch.einsum("i,j->ij", t, freqs)
    return torch.repeat_interleave(f, 2, dim=-1).to(dtype)


def _apply_rope(freqs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    cos, sin = (freqs.cos(), freqs.sin())
    return t * cos + _rotate_half(t) * sin


class AlternatingLayer(nn.Module):
    def __init__(self, dim=384, heads=8):
        super().__init__()
        from runtime.models.components.decoder_attention import TemporalAttention

        self.norm_q, self.norm_kv = (nn.LayerNorm(dim), nn.LayerNorm(dim))
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=0.0, batch_first=True)
        self.temporal = TemporalAttention(dim, heads)

    def forward(self, special, patches, batch, frames):
        count, dim = special.shape[1:]
        q, kv = (self.norm_q(special), self.norm_kv(patches))
        attention, _ = self.cross_attn(q, kv, kv, need_weights=False)
        special = special + attention
        tokens = special.view(batch, frames, count, dim).reshape(batch, frames * count, dim)
        tokens = self.temporal(tokens, frames, count)
        return tokens.reshape(batch, frames, count, dim).reshape(batch * frames, count, dim)


class MemoryAlternatingEncoder(nn.Module):
    def __init__(self, dim=384):
        super().__init__()
        self.hand_token = nn.Parameter(torch.empty(1, 2, dim))
        self.joint_token = nn.Parameter(torch.empty(1, 42, dim))
        self.register_token = nn.Parameter(torch.empty(1, 4, dim))
        self.readout_qnorm, self.readout_kvnorm = (nn.LayerNorm(dim), nn.LayerNorm(dim))
        self.readout_attn = nn.MultiheadAttention(dim, 8, dropout=0.0, batch_first=True)
        self.layers = nn.ModuleList([AlternatingLayer(dim) for _ in range(4)])
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x, batch, frames, height, width, slots):
        dim, combined = (x.shape[-1], batch * frames)
        hand = (self.hand_token + slots.view(1, 2, dim)).expand(combined, -1, -1)
        special = torch.cat(
            (
                hand,
                self.joint_token.expand(combined, -1, -1),
                self.register_token.expand(combined, -1, -1),
            ),
            dim=1,
        )
        special = special.view(batch, frames, 48, dim).reshape(combined, 48, dim)
        for layer in self.layers:
            special = layer(special, x, batch, frames)
        special = self.final_norm(special).view(batch, frames, 48, dim)
        hand = special[:, :, :2, :].view(batch, frames, 2, 1, dim).mean(dim=3)
        joint = special[:, :, 2:44, :].reshape(combined, 42, dim)
        q, kv = (self.readout_qnorm(joint), self.readout_kvnorm(x))
        _, probabilities = self.readout_attn(
            q, kv, kv, need_weights=True, average_attn_weights=True
        )
        anchors = self._soft_argmax(probabilities, height, width).reshape(batch, frames, 2, 21, 2)
        return (hand, anchors, probabilities.reshape(batch, frames, 2, 21, height, width))

    @staticmethod
    def _soft_argmax(attn_w: torch.Tensor, H: int, W: int) -> torch.Tensor:
        BF, n_q, _ = attn_w.shape
        hm = attn_w.reshape(BF, n_q, H, W)
        us = torch.linspace(0.0, 1.0, W, device=attn_w.device, dtype=attn_w.dtype)
        vs = torch.linspace(0.0, 1.0, H, device=attn_w.device, dtype=attn_w.dtype)
        wsum = hm.sum(dim=(-1, -2)).clamp(min=1e-06)
        u = (hm.sum(dim=-2) * us).sum(dim=-1) / wsum
        v = (hm.sum(dim=-1) * vs).sum(dim=-1) / wsum
        return torch.stack([u, v], dim=-1)
