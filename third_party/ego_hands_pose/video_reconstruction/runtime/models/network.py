import math

import torch
from torch import nn

from ..camera import calibrated_camera_rays
from .components.rays import normalized_grid
from .network_base import ReconstructionBase


class PinholeCameraHead(nn.Module):
    def __init__(self, dim, hidden=384, focal_min=0.05, focal_max=10.0):
        super().__init__()
        self.focal_min, self.focal_max = (float(focal_min), float(focal_max))
        self.tokens = nn.Sequential(nn.Linear(dim + 2, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.attention = nn.Linear(hidden, 1)
        self.readout = nn.Sequential(nn.Linear(hidden, 192), nn.GELU(), nn.Linear(192, 4))

    def forward(self, features):
        features = features.float().mean(1)
        b, h, w, _ = features.shape
        grid = normalized_grid(h, w, device=features.device, dtype=features.dtype)
        tokens = self.tokens(torch.cat([features, grid.expand(b, -1, -1, -1)], -1)).flatten(1, 2)
        pooled = (tokens * self.attention(tokens).softmax(1)).sum(1)
        params = self.readout(pooled)
        focal = (
            math.log(self.focal_min)
            + params[:, :2].sigmoid() * math.log(self.focal_max / self.focal_min)
        ).exp()
        center = 0.5 + params[:, 2:]
        xy = (grid - center[:, None, None]) / focal[:, None, None]
        raw = torch.cat([xy, torch.ones_like(xy[..., :1])], -1)
        return (raw, dict(focal=focal, center=center, valid=torch.isfinite(params).all(-1)))


class ReconstructionNetwork(ReconstructionBase):
    def __init__(self, backbone, mano):
        super().__init__(backbone, mano)
        self.camera_head = PinholeCameraHead(backbone.output_dim)

    def _decode(self, features, batch):
        k = getattr(batch, "geometry_intrinsics", None)
        raw, fit = (
            self.camera_head(features)
            if k is None
            else calibrated_camera_rays(features, k, batch.image_size)
        )
        if not bool(fit["valid"].all()):
            raise ValueError("Nonfinite camera parameters")
        return self._decode_outputs(raw, self.decoder(features, raw), batch, camera_fit=fit)
