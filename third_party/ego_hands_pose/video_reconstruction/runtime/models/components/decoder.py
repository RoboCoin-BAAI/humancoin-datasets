from dataclasses import dataclass

import torch


@dataclass(slots=True)
class DecoderOutput:
    hand_tokens: torch.Tensor
    anchors_2d: torch.Tensor
    joint_heatmaps: torch.Tensor
