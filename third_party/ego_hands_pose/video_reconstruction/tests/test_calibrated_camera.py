"""Calibrated rays, window transport and fixed-camera postprocessing contracts."""

import copy
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from runtime.accelerated_inference import (
    batch_for,
    combine_window_batches,
    iter_window_predictions,
)
from runtime.camera import calibrated_camera_rays, validate_intrinsics
from runtime.canvas_padding import resize_plan
from runtime.exact_kernel_runtime import ExactKernelRuntime
from runtime.fp8_runtime import decode_without_cpu_read, warm_protected_shapes
from runtime.global_consistency import apply_global_consistency
from runtime.models.components.readout import raw_bearings, solve_translation
from runtime.models.components.sequence_fusion import sequence_windows
from runtime.models.network import ReconstructionNetwork
from runtime.pipeline import Predictor

ROOT = Path(__file__).resolve().parents[1]
K = np.array([[500.0, 0, 336], [0, 510, 240], [0, 0, 1]])


@pytest.mark.parametrize(
    "container", [lambda k: k, lambda k: {"K": k}, lambda k: {"intrinsics": k}, json.dumps]
)
def test_input_formats(container):
    np.testing.assert_array_equal(validate_intrinsics(container(K.tolist())), K)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((3, 3)),
        np.eye(2),
        [np.eye(3), np.eye(3)],
        [[1, 1, 0], [0, 1, 0], [0, 0, 1]],
        np.diag([-1, 1, 1]),
        np.full((3, 3), np.nan),
        {"K": None},
    ],
)
def test_invalid_intrinsics(bad):
    with pytest.raises(ValueError, match="intrinsics"):
        validate_intrinsics(bad)


@pytest.mark.parametrize(
    "mode,source",
    [("center_crop", [1280, 720]), ("center_crop", [640, 960]), ("long_edge_pad", [1280, 720])],
)
def test_rays_project_to_feature_cell_centres_after_preprocessing(mode, source):
    plan = resize_plan(source, mode, width=672, height=480)
    width, height = plan["size"]
    source_k = K.copy()
    source_k[:2, 2] = np.array(source) / 2
    canvas_k = np.asarray(plan["source_to_canvas"]) @ source_k
    rays, fit = calibrated_camera_rays(
        torch.zeros(1, 21, height // 32, width // 32, 2), canvas_k, torch.tensor([[height, width]])
    )
    uv = rays[0, ..., :2].numpy() * canvas_k[[0, 1], [0, 1]] + canvas_k[:2, 2]
    yy, xx = np.meshgrid(
        (np.arange(height // 32) + 0.5) * 32, (np.arange(width // 32) + 0.5) * 32, indexing="ij"
    )
    np.testing.assert_allclose(uv, np.stack((xx, yy), -1), atol=5e-5)
    np.testing.assert_array_equal(rays[..., 2], 1)
    np.testing.assert_allclose(fit["focal"], canvas_k[[0, 1], [0, 1]][None] / [width, height])
    # Cell-centre interpolation remains exact at the canvas borders.
    anchors = torch.tensor([[[0.0, 0.0], [1.0, 1.0], [0.37, 0.62]]])
    bearings = raw_bearings(rays, anchors)
    expected = (
        anchors * torch.tensor([width, height]) - torch.tensor(canvas_k[:2, 2])
    ) / torch.tensor(canvas_k[[0, 1], [0, 1]])
    torch.testing.assert_close(bearings.double(), expected, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize(
    "mode,source,target,affine",
    [
        (
            "center_crop",
            [1280, 720],
            [672, 480],
            [[2 / 3, 0, -136 * 2 / 3 - 1 / 6], [0, 2 / 3, -1 / 6], [0, 0, 1]],
        ),
        (
            "center_crop",
            [640, 960],
            [480, 480],
            [[0.75, 0, -0.125], [0, 0.75, -120.125], [0, 0, 1]],
        ),
        (
            "long_edge_pad",
            [1280, 720],
            [832, 480],
            [[0.65, 0, -0.175], [0, 0.65, 5.825], [0, 0, 1]],
        ),
    ],
)
def test_source_camera_transform_includes_crop_half_pixel_and_padding(mode, source, target, affine):
    plan = resize_plan(source, mode, width=target[0], height=target[1])
    assert plan["size"] == target
    np.testing.assert_allclose(plan["source_to_canvas"], affine, atol=1e-12)
    source_point = np.array([0.2, -0.1, 1.0])
    source_pixel = K @ source_point
    canvas_pixel = np.asarray(plan["source_to_canvas"]) @ K @ source_point
    np.testing.assert_allclose(canvas_pixel, np.asarray(affine) @ source_pixel, atol=1e-12)


def test_invalid_request_k_is_rejected_before_model_loading():
    # No device or model is initialized for a malformed request.
    predictor = Predictor.__new__(Predictor)
    with pytest.raises(ValueError, match="intrinsics"):
        predictor.run("unused.mp4", "unused-output", geometry_intrinsics=np.eye(2))


@pytest.mark.parametrize("decode", [ReconstructionNetwork._decode, decode_without_cpu_read])
@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_both_decoders_bypass_camera_head_and_do_not_retain_k(decode, batch_size):
    features = torch.zeros(batch_size, 21, 15, 21, 3)
    seen = {}

    def head(_):
        seen["head_calls"] = seen.get("head_calls", 0) + 1
        return torch.ones(batch_size, 15, 21, 3), {
            "valid": torch.ones(batch_size, dtype=torch.bool)
        }

    def decoder(_, rays):
        seen["decoder_rays"] = rays
        return "decoded"

    def outputs(rays, decoded, batch, *, camera_fit):
        assert decoded == "decoded" and rays is seen["decoder_rays"]
        return rays, camera_fit

    net = SimpleNamespace(
        camera_head=head, decoder=decoder, _decode_outputs=outputs, _deferred_camera_checks=[]
    )
    batch = SimpleNamespace(image_size=torch.tensor([[480, 672]]).expand(batch_size, -1))
    results = []
    for k in [K, K * np.array([[1.2, 1, 1], [1, 1.2, 1], [1, 1, 1]]), None, K]:
        batch.geometry_intrinsics = k
        results.append(decode(net, features, batch)[0])
    assert seen["head_calls"] == 1
    assert not torch.equal(results[0], results[1])
    torch.testing.assert_close(results[0], results[-1], rtol=0, atol=0)


def test_translation_solver_uses_the_calibrated_rays():
    index = torch.arange(21, dtype=torch.float32)
    points = torch.stack(
        ((index % 5 - 2) * 0.015, (index // 5 - 2) * 0.018, (index % 3 - 1) * 0.005), -1
    )
    canonical = points[None, None, None].expand(1, 1, 2, -1, -1)
    truth = torch.tensor([0.04, -0.02, 0.6]).expand(1, 1, 2, 3)
    placed = canonical + truth[..., None, :]
    anchors = (
        (
            placed[..., :2] / placed[..., 2:] * torch.tensor(K[[0, 1], [0, 1]])
            + torch.tensor(K[:2, 2])
        )
        / torch.tensor([672, 480])
    ).float()
    rays, _ = calibrated_camera_rays(torch.zeros(1, 21, 15, 21, 3), K, torch.tensor([[480, 672]]))
    camera = torch.zeros(1, 1, 2, 3)
    camera[..., 2] = np.log(0.6)
    actual = solve_translation(
        canonical,
        anchors,
        camera,
        torch.tensor([[[480, 672]]]),
        rays=rays,
        fallback_bearing=torch.zeros(1, 1, 2, 2),
    )
    torch.testing.assert_close(actual, truth, rtol=1e-5, atol=1e-6)


@dataclass
class Prediction:
    value: torch.Tensor


@pytest.mark.parametrize("frames", [9, 81, 85, 243, 1000])
@pytest.mark.parametrize("size", [1, 4, 16])
def test_grouped_windows_and_tail_keep_fixed_k(frames, size):
    latent = torch.zeros(48, 1 + (frames + 2) // 4, 2, 2)
    windows = sequence_windows(frames, 81, 40)
    calls = []

    def net(batch):
        b = batch.latent_mean.shape[0]
        np.testing.assert_array_equal(batch.geometry_intrinsics, np.broadcast_to(K, (b, 3, 3)))
        calls.append(b)
        return Prediction(torch.zeros(b, 1))

    actual = list(
        iter_window_predictions(
            net, windows, latent, torch.device("cpu"), 480, 672, size, geometry_intrinsics=K
        )
    )
    assert [window for window, _ in actual] == windows
    assert max(calls) <= size
    if frames == 1000 and size > 1:
        assert max(calls) == size and calls[-1] == 1


def test_mixed_camera_modes_cannot_be_combined():
    latent = torch.zeros(48, 22, 2, 2)
    with pytest.raises(ValueError, match="cannot mix"):
        combine_window_batches(
            [
                batch_for(0, 81, latent, "cpu"),
                batch_for(0, 81, latent, "cpu", geometry_intrinsics=K),
            ]
        )


@pytest.mark.parametrize("path", ["bf16", "fp8", "h100"])
def test_warmup_routes_transport_k_and_separate_modes(monkeypatch, path):
    import runtime.accelerated_inference as accelerated
    import runtime.fp8_modules as modules
    from runtime.hopper import prepare

    calls = []

    def net(batch):
        assert getattr(batch, "geometry_intrinsics", None) is not None
        calls.append(batch.latent_mean.shape[0])

    net._deferred_camera_checks = []

    def infer(*args, **kwargs):
        np.testing.assert_array_equal(kwargs["geometry_intrinsics"], K)
        return {"latent_mean": np.zeros((48, 22, 30, 30), np.float32)}, {}

    monkeypatch.setattr(accelerated, "infer_video", infer)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    monkeypatch.setattr(modules, "populate_rope_phases", lambda *args: None)
    state = ExactKernelRuntime()
    vae = SimpleNamespace(dtype=torch.float32)
    predictor = Predictor.__new__(Predictor)
    predictor.net, predictor.vae, predictor.device = net, vae, torch.device("cpu")
    predictor.acceleration_evidence = {}
    predictor.config = {"window_batch_size": 4}
    predictor._warm, predictor._calibrated_warm = set(), set()
    state.vae, state.launch_defaults = vae, {}
    kwargs = dict(
        width=480,
        height=480,
        frames=243,
        resize_mode="center_crop",
        long_edge=832,
        geometry_intrinsics=K,
    )
    if path == "bf16":
        predictor._prepare(
            "video",
            {"size": [480, 480], "resize_mode": "center_crop"},
            243,
            "bf16",
            state,
            lambda *args: None,
            K,
        )
        assert predictor._calibrated_warm and not predictor._warm
    elif path == "fp8":
        warm_protected_shapes(net, vae, "video", torch.device("cpu"), **kwargs)
    else:
        prepare(predictor, state, "video", progress=lambda *args: None, **kwargs)
        assert state.camera_coverage(K) and not state.camera_coverage(None)
    assert calls


class Geometry:
    def __call__(self, orient, pose, beta, translation):
        index = torch.arange(21, dtype=beta.dtype, device=beta.device)
        base = torch.stack(
            ((index % 5 - 2) * 0.015, (index // 5 - 2) * 0.018, (index % 3 - 1) * 0.005), -1
        )
        points = base * (1 + beta[..., :1, None] * 0.03) + translation[..., None, :]
        return points, torch.cat((points, points + 0.001), -2)


def test_consistency_keeps_calibration_and_raw_arrays():
    count = 5
    orient, pose, beta = (
        np.zeros((count, 2, 3), np.float32),
        np.zeros((count, 2, 45), np.float32),
        np.zeros((count, 2, 10), np.float32),
    )
    translation = np.tile([[[-0.1, -0.04, 0.6], [0.08, -0.03, 0.7]]], (count, 1, 1)).astype(
        np.float32
    )
    joints, vertices = Geometry()(
        *(torch.from_numpy(x)[None] for x in (orient, pose, beta, translation))
    )
    joints, vertices = joints[0].numpy(), vertices[0].numpy()
    transform = np.array([[0.5, 0, -20], [0, 0.5, 10], [0, 0, 1.0]])
    source_k = np.array([[800, 0, 640], [0, 820, 360], [0, 0, 1.0]])
    canvas_k = transform @ source_k
    camera = np.broadcast_to(canvas_k, (count, 3, 3)).copy()
    anchors = (joints[..., :2] / joints[..., 2:] * canvas_k[[0, 1], [0, 1]] + canvas_k[:2, 2]) / [
        672,
        480,
    ]
    arrays = dict(
        global_orient=orient,
        hand_pose=pose,
        betas=beta,
        translation=translation,
        joints_3d=joints,
        vertices=vertices,
        anchors_2d=anchors,
        predicted_intrinsics=camera,
        overlay_intrinsics=camera.copy(),
        existence_prob=np.ones((count, 2)),
        visibility_prob=np.ones((count, 2)),
        frame_indices=np.arange(count),
        timestamps=np.arange(count) / 30,
    )
    arrays["predicted_intrinsics"][:, 0, 0] *= 1.1
    saved = copy.deepcopy(arrays)
    metadata = dict(
        architecture="parametric_pinhole",
        size=[672, 480],
        source_size=[1280, 720],
        source_to_canvas=transform,
        overlay_camera="calibrated_parametric_pinhole",
    )
    output, report = apply_global_consistency(
        arrays, metadata, Geometry(), fixed_intrinsics_source=source_k
    )
    np.testing.assert_array_equal(
        output["predicted_intrinsics"], np.broadcast_to(canvas_k, (count, 3, 3))
    )
    np.testing.assert_array_equal(output["global_intrinsics_source"][0], source_k)
    assert report["camera_source"] == "calibration" and report["camera_aggregated"] is False
    for name in saved:
        np.testing.assert_array_equal(arrays[name], saved[name])


def test_cli_rejects_geometry_k_for_rerender():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "infer.py"),
            "input.mp4",
            "--rerender",
            "result",
            "--geometry-camera-json",
            "camera.json",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2 and "cannot be used with --rerender" in result.stderr
