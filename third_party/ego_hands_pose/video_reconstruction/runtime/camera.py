"""Validated pixel intrinsics and calibrated decoder rays."""

import json

import numpy as np
import torch

from .models.components.rays import normalized_grid


def validate_intrinsics(value):
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, dict):
        value = value.get("intrinsics", value.get("K"))
    try:
        k = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("Camera intrinsics must be a numeric 3x3 pinhole matrix") from error
    if (
        k.shape != (3, 3)
        or not np.isfinite(k).all()
        or (k[[0, 1], [0, 1]] <= 0).any()
        or not np.allclose(k[2], [0, 0, 1])
        or not np.allclose(k[[0, 1], [1, 0]], 0)
    ):
        raise ValueError(
            "Camera intrinsics must be a finite 3x3 pinhole matrix with positive focal "
            "lengths, zero skew and bottom row [0,0,1]"
        )
    return k.copy()


def calibrated_camera_rays(features, intrinsics, image_size):
    """Build z=1 rays from canvas K already validated at the request boundary."""
    b, _, h, w, _ = features.shape
    k = torch.as_tensor(intrinsics, device=features.device, dtype=torch.float32)
    if k.shape == (3, 3):
        k = k[None].expand(b, -1, -1)
    if k.shape != (b, 3, 3) or image_size.shape != (b, 2):
        raise ValueError("Calibrated K and image size must match the window batch")
    scale = image_size.to(device=features.device, dtype=torch.float32).flip(-1)
    focal = k[:, [0, 1], [0, 1]] / scale
    center = k[:, :2, 2] / scale
    grid = normalized_grid(h, w, device=features.device, dtype=torch.float32)
    xy = (grid - center[:, None, None]) / focal[:, None, None]
    rays = torch.cat((xy, torch.ones_like(xy[..., :1])), -1)
    return rays, dict(
        focal=focal, center=center, valid=torch.ones(b, device=features.device, dtype=torch.bool)
    )
