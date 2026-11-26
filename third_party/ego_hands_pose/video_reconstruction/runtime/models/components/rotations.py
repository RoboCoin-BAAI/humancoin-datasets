from __future__ import annotations

import torch
import torch.nn.functional as F

from .precision import fp32_geometry


def _sqrt_positive_part(value: torch.Tensor) -> torch.Tensor:
    floor = torch.finfo(value.dtype).eps
    return torch.where(value > 0, torch.sqrt(value.clamp_min(floor)), torch.zeros_like(value))


@fp32_geometry
def rotation_6d_to_matrix(d6: torch.Tensor, eps: float = 1e-08) -> torch.Tensor:
    if d6.shape[-1] != 6:
        raise ValueError(f"Expected last dimension 6, got {d6.shape}")
    a1, a2 = (d6[..., :3], d6[..., 3:])
    b1 = F.normalize(a1, dim=-1, eps=eps)
    first_axis = torch.zeros_like(b1)
    first_axis[..., 0] = 1
    b1 = torch.where(torch.linalg.vector_norm(a1, dim=-1, keepdim=True) > eps, b1, first_axis)
    residual = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    basis = F.one_hot(b1.abs().argmin(-1), num_classes=3).to(b1.dtype)
    perpendicular = F.normalize(basis - (basis * b1).sum(-1, keepdim=True) * b1, dim=-1)
    b2 = torch.where(
        torch.linalg.vector_norm(residual, dim=-1, keepdim=True) > eps,
        F.normalize(residual, dim=-1, eps=eps),
        perpendicular,
    )
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


@fp32_geometry
def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.vector_norm(axis_angle, dim=-1, keepdim=True)
    x, y, z = axis_angle.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(
        *axis_angle.shape[:-1], 3, 3
    )
    eye = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    first = torch.sinc(theta / torch.pi)[..., None]
    second = 0.5 * torch.sinc(theta / (2 * torch.pi)).square()[..., None]
    skew_squared = (
        axis_angle[..., :, None] * axis_angle[..., None, :]
        - axis_angle.square().sum(-1)[..., None, None] * eye
    )
    return eye + first * skew + second * skew_squared


@fp32_geometry
def matrix_to_axis_angle(matrix: torch.Tensor, eps: float = 1e-06) -> torch.Tensor:
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (...,3,3), got {matrix.shape}")
    m00, m01, m02 = (matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2])
    m10, m11, m12 = (matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2])
    m20, m21, m22 = (matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2])
    q_abs = _sqrt_positive_part(
        torch.stack(
            (1 + m00 + m11 + m22, 1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22), -1
        )
    )
    candidates = torch.stack(
        (
            torch.stack((q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01), -1),
            torch.stack((m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20), -1),
            torch.stack((m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21), -1),
            torch.stack((m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2), -1),
        ),
        -2,
    )
    candidates = candidates / (2 * q_abs.clamp_min(0.1)[..., None])
    choice = F.one_hot(q_abs.argmax(-1), num_classes=4).to(matrix.dtype)
    quaternion = (candidates * choice[..., None]).sum(-2)
    quaternion = F.normalize(quaternion, dim=-1, eps=eps)
    quaternion = torch.where(quaternion[..., :1] < 0, -quaternion, quaternion)
    vector = quaternion[..., 1:]
    vector_norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    half_angle = torch.atan2(vector_norm, quaternion[..., :1])
    scale = 2 * half_angle / vector_norm.clamp_min(eps)
    scale = torch.where(vector_norm < 0.0001, 2 + vector_norm.square() / 3, scale)
    return vector * scale
