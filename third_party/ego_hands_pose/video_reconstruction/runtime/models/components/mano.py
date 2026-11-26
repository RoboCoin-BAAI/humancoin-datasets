from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .precision import fp32_geometry
from .rotations import axis_angle_to_matrix


class BatchedMANO(nn.Module):
    def __init__(self, model_path: str) -> None:
        super().__init__()
        import smplx

        root = Path(model_path)
        self.left = smplx.MANO(
            str(root / "MANO_LEFT.npz"), is_rhand=False, use_pca=False, ext="npz"
        )
        self.right = smplx.MANO(
            str(root / "MANO_RIGHT.npz"), is_rhand=True, use_pca=False, ext="npz"
        )
        self.requires_grad_(False)

    @fp32_geometry
    def forward(
        self,
        global_orient: torch.Tensor,
        hand_pose: torch.Tensor,
        betas: torch.Tensor,
        translation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from smplx.lbs import lbs

        shape = global_orient.shape[:-1]
        outputs = []
        for hand, layer in enumerate((self.left, self.right)):
            orient = global_orient[..., hand, :].reshape(-1, 3)
            pose = hand_pose[..., hand, :].reshape(-1, 45)
            full_pose = torch.cat((orient, pose), -1) + layer.pose_mean
            rotations = axis_angle_to_matrix(full_pose.reshape(-1, 16, 3))
            vertices, raw = lbs(
                betas[..., hand, :].reshape(-1, 10),
                rotations,
                layer.v_template,
                layer.shapedirs,
                layer.posedirs,
                layer.J_regressor,
                layer.parents,
                layer.lbs_weights,
                pose2rot=False,
            )
            transl = translation[..., hand, :].reshape(-1, 1, 3)
            vertices, raw = (vertices + transl, raw + transl)
            ids = (744, 320, 443, 554, 671)
            tips = vertices[:, ids]
            joints = torch.stack(
                (
                    raw[:, 0],
                    raw[:, 13],
                    raw[:, 14],
                    raw[:, 15],
                    tips[:, 0],
                    raw[:, 1],
                    raw[:, 2],
                    raw[:, 3],
                    tips[:, 1],
                    raw[:, 4],
                    raw[:, 5],
                    raw[:, 6],
                    tips[:, 2],
                    raw[:, 10],
                    raw[:, 11],
                    raw[:, 12],
                    tips[:, 3],
                    raw[:, 7],
                    raw[:, 8],
                    raw[:, 9],
                    tips[:, 4],
                ),
                dim=1,
            )
            outputs.append((joints, vertices))
        joints = torch.stack([value[0] for value in outputs], dim=-3).reshape(*shape[:-1], 2, 21, 3)
        vertices = torch.stack([value[1] for value in outputs], dim=-3).reshape(
            *shape[:-1], 2, -1, 3
        )
        return (joints, vertices)
