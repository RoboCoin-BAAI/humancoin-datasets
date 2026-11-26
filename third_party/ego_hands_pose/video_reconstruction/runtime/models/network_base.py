import torch
from torch import nn
from torch.nn import functional as F

from runtime.data.schema import SequencePrediction

from .components.readout import SpatialTemporalDecoder, readout_head, solve_translation
from .components.rotations import matrix_to_axis_angle, rotation_6d_to_matrix


class ReconstructionBase(nn.Module):
    def __init__(self, backbone, mano):
        super().__init__()
        self.backbone, self.mano = (backbone, mano)
        self.decoder = SpatialTemporalDecoder(backbone.output_dim)
        self.hand_fuse = nn.Sequential(nn.LayerNorm(384), nn.Linear(384, 384), nn.GELU())
        self.head_global_orient = readout_head(6)
        self.head_hand_pose = readout_head(90)
        self.head_cam_trans = readout_head(3)
        self.head_betas_hand = readout_head(10, norm=True)
        self.head_exists_2d = readout_head(1)
        self.head_exists_3d = readout_head(1)
        self.head_fallback_bearing = readout_head(2)

    def forward(self, batch):
        features = self.backbone(batch.latent_mean)
        with torch.autocast(features.device.type, enabled=False):
            return self._decode(features.float(), batch)

    @staticmethod
    def _temporal_expand(
        value: torch.Tensor, output_frames: int, temporal_offset: torch.Tensor | None
    ) -> torch.Tensor:
        shape = value.shape
        flat = value.flatten(2).transpose(1, 2)
        supported_frames = 4 * (shape[1] - 1) + 1
        flat = F.interpolate(
            flat, size=supported_frames, mode="linear", align_corners=True
        ).transpose(1, 2)
        if temporal_offset is None:
            temporal_offset = torch.zeros(shape[0], device=value.device, dtype=torch.long)
        else:
            temporal_offset = temporal_offset.to(device=value.device, dtype=torch.long)
        gather = temporal_offset[:, None] + torch.arange(output_frames, device=value.device)[None]
        gather = gather[..., None].expand(-1, -1, flat.shape[-1])
        flat = flat.gather(1, gather)
        return flat.reshape(shape[0], output_frames, *shape[2:])

    def _decode_outputs(self, raw, decoded, batch, *, camera_fit):

        def expand(v):
            return self._temporal_expand(v, batch.rgb_frames, batch.temporal_offset)

        hand = expand(decoded.hand_tokens)
        hand = hand + self.hand_fuse(hand)
        anchors = expand(decoded.anchors_2d)
        pose = torch.cat((self.head_global_orient(hand), self.head_hand_pose(hand)), -1).unflatten(
            -1, (16, 6)
        )
        aa = matrix_to_axis_angle(rotation_6d_to_matrix(pose))
        go, hp = (aa[..., 0, :], aa[..., 1:, :].flatten(-2))
        betas = self.head_betas_hand(hand.mean(dim=1))[:, None].expand(-1, hand.shape[1], -1, -1)
        camera_raw = self.head_cam_trans(hand)
        ex2 = self.head_exists_2d(hand).squeeze(-1)
        ex3 = self.head_exists_3d(hand).squeeze(-1)
        canonical, _ = self.mano(go, hp, betas, torch.zeros_like(camera_raw))
        size = batch.image_size
        if size.ndim == 2:
            size = size[:, None].expand(-1, hand.shape[1], -1)
        fallback_bearing = self.head_fallback_bearing(hand)
        translation = solve_translation(
            canonical, anchors, camera_raw, size, rays=raw, fallback_bearing=fallback_bearing
        )
        return SequencePrediction(
            global_orient=go,
            hand_pose=hp,
            betas=betas,
            translation=translation,
            anchors_2d=anchors,
            existence_prob=ex2.sigmoid(),
            visibility_prob=ex3.sigmoid(),
            camera_fit=camera_fit,
            joint_heatmaps=decoded.joint_heatmaps,
            heatmap_frame_indices=torch.arange(
                decoded.joint_heatmaps.shape[1], device=anchors.device
            )[None].expand(anchors.shape[0], -1)
            * 4
            - batch.temporal_offset.to(anchors.device)[:, None],
        )
