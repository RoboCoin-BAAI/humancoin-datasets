import torch

from runtime.data.schema import SequencePrediction

from .precision import fp32_geometry
from .rotations import axis_angle_to_matrix, matrix_to_axis_angle


def sequence_windows(total, window, stride):
    if total < 1 or window < 1 or (not 1 <= stride <= window):
        raise ValueError("require total/window > 0 and 1 <= stride <= window")
    last = max(0, total - window)
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return [(start, min(window, total - start)) for start in starts]


def latent_window(start, count, latent_frames):
    if start < 0 or count < 1:
        raise ValueError("invalid RGB window")
    begin, offset = divmod(start, 4)
    end = begin + 1 + (offset + count - 1 + 3) // 4
    if end > latent_frames:
        raise ValueError("RGB window exceeds latent support")
    return (slice(begin, end), offset)


@fp32_geometry
def project_rotation_mean(matrix):
    u, _, vh = torch.linalg.svd(matrix)
    signs = torch.ones_like(matrix[..., 0, :])
    signs[..., -1] = torch.linalg.det(u @ vh)
    return u * signs[..., None, :] @ vh


class SequenceFusion:
    fields = ("betas", "translation", "anchors_2d", "existence_prob", "visibility_prob")

    def __init__(self, total, pose_mean, storage_device="cpu"):
        if total < 1:
            raise ValueError("empty sequence")
        self.total = total
        self.storage_device = torch.device(storage_device)
        self.pose_mean = pose_mean.detach().float().to(self.storage_device).reshape(2, 16, 3)
        self.counts = torch.zeros(total, device=self.storage_device)
        self.sums = {}
        self.windows = set()

    @torch.no_grad()
    def add(self, start, prediction):
        count = prediction.translation.shape[1]
        if prediction.translation.shape[0] != 1 or start < 0 or start + count > self.total:
            raise ValueError("window outside sequence or batch size is not one")
        if (start, count) in self.windows:
            raise ValueError("duplicate window")
        self.windows.add((start, count))
        index = torch.arange(count, dtype=torch.float32, device=self.storage_device)
        weight = torch.minimum(index + 1, count - index)
        values = {
            name: getattr(prediction, name)[0].detach().float().to(self.storage_device)
            for name in self.fields
        }
        aa = (
            torch.cat((prediction.global_orient, prediction.hand_pose), -1)[0]
            .detach()
            .float()
            .to(self.storage_device)
        )
        values["rotations"] = axis_angle_to_matrix(aa.reshape(count, 2, 16, 3) + self.pose_mean)
        for name, value in values.items():
            if self.storage_device.type == "cpu" and (not torch.isfinite(value).all()):
                raise ValueError("nonfinite prediction cannot be fused")
            if name not in self.sums:
                self.sums[name] = torch.zeros(
                    self.total, *value.shape[1:], device=self.storage_device
                )
            self.sums[name][start : start + count] += value * weight.reshape(
                count, *[1] * (value.ndim - 1)
            )
        self.counts[start : start + count] += weight

    @torch.inference_mode()
    def finish(self, mano, device, chunk_size=512):
        if (self.counts == 0).any() or chunk_size < 1:
            raise ValueError("uncovered RGB frames or invalid chunk size")
        values = {
            name: value / self.counts.reshape(self.total, *[1] * (value.ndim - 1))
            for name, value in self.sums.items()
        }
        rotation = project_rotation_mean(values.pop("rotations"))
        aa = (matrix_to_axis_angle(rotation) - self.pose_mean).reshape(self.total, 2, 48)
        values["global_orient"], values["hand_pose"] = (aa[..., :3], aa[..., 3:])
        joints, vertices = ([], [])
        for start in range(0, self.total, chunk_size):
            args = [
                values[name][None, start : start + chunk_size].to(device)
                for name in ("global_orient", "hand_pose", "betas", "translation")
            ]
            j, v = mano(*args)
            joints.append(j.to(self.storage_device))
            vertices.append(v.to(self.storage_device))
        values = {name: value.unsqueeze(0) for name, value in values.items()}
        return SequencePrediction(
            **values, joints_3d=torch.cat(joints, 1), vertices=torch.cat(vertices, 1)
        )
