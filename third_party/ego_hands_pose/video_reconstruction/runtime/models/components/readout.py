import torch
from torch import nn
from torch.nn import functional as F

from runtime.vendor.attention.memory_encoder import MemoryAlternatingEncoder

from .decoder import DecoderOutput
from .precision import fp32_geometry


class RayEncoding(nn.Module):
    def __init__(self, dim=384):
        super().__init__()
        self.ray_proj = nn.Sequential(nn.Linear(32, dim), nn.GELU(), nn.Linear(dim, dim))
        self.register_buffer("freq_bands", 2.0 ** torch.arange(8).float())

    def forward(self, rays):
        z = rays[..., 2].abs().clamp_min(1e-06)
        az = torch.atan2(rays[..., 0], z)[..., None] * self.freq_bands
        el = torch.atan2(rays[..., 1], z)[..., None] * self.freq_bands
        return self.ray_proj(torch.cat((az.sin(), az.cos(), el.sin(), el.cos()), -1))


class SpatialTemporalDecoder(nn.Module):
    def __init__(self, backbone_dim=3072):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(backbone_dim, 384), nn.LayerNorm(384))
        self.spatial_pe_grid = nn.Parameter(torch.randn(1, 384, 16, 16) * 0.02)
        self.slot_embed = nn.Parameter(torch.randn(2, 384) * 0.02)
        self.ray_pe = RayEncoding()
        self.alternating_encoder = MemoryAlternatingEncoder(384)

    def forward(self, features, rays):
        with torch.autocast(features.device.type, enabled=False):
            b, t, h, w, _ = features.shape
            x = self.input_proj(features.float()).reshape(b, t, h * w, 384)
            pe = F.interpolate(self.spatial_pe_grid, (h, w), mode="bilinear", align_corners=True)
            x = x + pe.flatten(2).transpose(1, 2)[:, None]
            x = x + self.ray_pe(rays.float()).reshape(b, 1, h * w, 384)
            outputs = self.alternating_encoder(
                x.reshape(b * t, h * w, 384), b, t, h, w, self.slot_embed
            )
            return DecoderOutput(*outputs)


def readout_head(output_dim, *, norm=False):
    layers = [nn.LayerNorm(384)] if norm else []
    return nn.Sequential(*layers, nn.Linear(384, 192), nn.GELU(), nn.Linear(192, output_dim))


def sample_raw_rays(rays, coordinates):
    b = rays.shape[0]
    grid = coordinates.reshape(b, -1, 1, 2)
    field = rays.permute(0, 3, 1, 2)
    h, w = field.shape[-2:]
    left = 2 * field[..., :1] - field[..., 1:2] if w > 1 else field[..., :1]
    right = 2 * field[..., -1:] - field[..., -2:-1] if w > 1 else field[..., -1:]
    field = torch.cat((left, field, right), -1)
    top = 2 * field[..., :1, :] - field[..., 1:2, :] if h > 1 else field[..., :1, :]
    bottom = 2 * field[..., -1:, :] - field[..., -2:-1, :] if h > 1 else field[..., -1:, :]
    field = torch.cat((top, field, bottom), -2)
    shape = grid.new_tensor([w, h])
    grid = (grid * shape + 1) / (shape + 2) * 2 - 1
    sampled = F.grid_sample(field, grid, align_corners=False, padding_mode="border")
    return sampled[..., 0].transpose(1, 2).reshape(*coordinates.shape[:-1], 3)


def raw_bearings(rays, coordinates):
    sample = sample_raw_rays(rays, coordinates)
    return sample[..., :2] / sample[..., 2:].abs().clamp_min(1e-06)


@fp32_geometry
def solve_translation(canonical, anchors, camera_raw, image_size, *, rays, fallback_bearing):
    depth = camera_raw[..., 2].exp()
    size = image_size[..., None, :]
    bearing = raw_bearings(rays, anchors)
    scale = size[..., 1:2].expand(*size.shape[:-1], 2)
    fallback_translation = torch.cat((fallback_bearing * depth[..., None], depth[..., None]), -1)
    z_raw = canonical[..., 2] + depth[..., None]
    z = z_raw.clamp_min(0.05)
    with torch.no_grad():
        voters = (z_raw > 0.05) & ((anchors > 0.02) & (anchors < 0.98)).all(-1)
    m = voters.to(z.dtype)
    inv_z = z.reciprocal()
    xy = (m[..., None] * inv_z[..., None] * (bearing - canonical[..., :2] * inv_z[..., None])).sum(
        -2
    )
    xy = xy / (m * inv_z.square()).sum(-1, keepdim=True).clamp_min(1e-08)
    candidate = torch.cat((xy, depth[..., None]), -1)
    with torch.no_grad():
        obs = bearing * scale[..., None, :]
        fitted = (canonical[..., :2] + xy[..., None, :]) / z[..., None] * scale[..., None, :]
        n = m.sum(-1)
        r2 = (fitted - obs).square().sum(-1)
        rms = (torch.where(voters, r2, 0.0).sum(-1) / n.clamp_min(1)).sqrt()
        hi = torch.where(voters[..., None], obs, -1000000000.0).amax(-2)
        lo = torch.where(voters[..., None], obs, 1000000000.0).amin(-2)
        threshold = (0.25 * torch.linalg.vector_norm(hi - lo, dim=-1)).clamp_min(15.0)
        insufficient = n < 6
        excessive = rms > threshold
        nonfinite = ~torch.isfinite(candidate).all(-1) | ~torch.isfinite(rms)
        fallback = insufficient | excessive | nonfinite
    return torch.where(fallback[..., None], fallback_translation, candidate)
