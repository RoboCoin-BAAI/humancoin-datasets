from __future__ import annotations

import torch


def normalized_grid(height: int, width: int, *, device=None, dtype=None) -> torch.Tensor:
    ys = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
    xs = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((xx, yy), dim=-1)
