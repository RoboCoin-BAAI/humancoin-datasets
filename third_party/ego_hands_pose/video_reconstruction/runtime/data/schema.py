from dataclasses import dataclass

import torch


@dataclass(slots=True)
class SequencePrediction:
    global_orient: torch.Tensor
    hand_pose: torch.Tensor
    betas: torch.Tensor
    translation: torch.Tensor
    anchors_2d: torch.Tensor
    existence_prob: torch.Tensor
    visibility_prob: torch.Tensor
    joints_3d: torch.Tensor | None = None
    vertices: torch.Tensor | None = None
    camera_fit: dict | None = None
    joint_heatmaps: torch.Tensor | None = None
    heatmap_frame_indices: torch.Tensor | None = None
