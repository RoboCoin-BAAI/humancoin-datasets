"""GPU execution policies and SM90 FP8 launch configuration."""

import json
from contextlib import contextmanager

import torch

from .runtime import ROOT


@contextmanager
def execution_policy(device):
    from .bf16_runtime import validated_device

    sm80 = validated_device(device)
    active = torch.cuda.get_device_capability(device) == (9, 0) or sm80
    flags = (
        torch.are_deterministic_algorithms_enabled(),
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    if active:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = sm80
        torch.backends.cudnn.deterministic = True
    try:
        yield
    finally:
        if active:
            torch.use_deterministic_algorithms(flags[0], warn_only=flags[3])
            torch.backends.cudnn.benchmark = flags[1]
            torch.backends.cudnn.deterministic = flags[2]


def prepare(
    predictor,
    state,
    source,
    *,
    width,
    height,
    frames,
    resize_mode,
    long_edge,
    progress,
    geometry_intrinsics=None,
):
    from .accelerated_inference import batch_for, combine_window_batches, infer_video, window_groups
    from .fp8_modules import ImplicitFP8Conv
    from .fp8_runtime import validate_deferred_camera
    from .kernel_compiled_qkv import CompiledQKV
    from .models.components.sequence_fusion import latent_window, sequence_windows

    if state.vae is not predictor.vae:
        state.suspend()
        state.__init__()
        state.vae = predictor.vae
    if not hasattr(state, "launch_defaults"):
        state.launch_defaults = {
            name: (conv.block, conv.num_warps, conv.num_stages, conv.maxnreg)
            for name, conv in predictor.vae.named_modules()
            if isinstance(conv, ImplicitFP8Conv)
        }
    for name, defaults in state.launch_defaults.items():
        conv = predictor.vae.get_submodule(name)
        conv.block, conv.num_warps, conv.num_stages, conv.maxnreg = defaults
    if width in (672, 832):
        choices = json.loads((ROOT / f"assets/acceleration/hopper_{width}.json").read_text())
        for name, choice in choices.items():
            conv = predictor.vae.get_submodule(name)
            if not isinstance(conv, ImplicitFP8Conv) or set(choice) != {
                "block",
                "num_warps",
                "num_stages",
                "maxnreg",
            }:
                raise ValueError("Invalid FP8 launch policy")
            conv.block = tuple(choice["block"])
            conv.num_warps, conv.num_stages, conv.maxnreg = (
                choice["num_warps"],
                choice["num_stages"],
                choice["maxnreg"],
            )
    signatures = set()
    for group, size in window_groups(sequence_windows(frames, 81, 40), 1 + (frames + 2) // 4, 16):
        selection, offset = latent_window(*group[0], 1 + (frames + 2) // 4)
        signatures.add((width, height, offset, group[0][1], selection.stop - selection.start, size))
    coverage = state.camera_coverage(geometry_intrinsics)
    missing = signatures - coverage
    if missing:
        progress(0.1, "预热 FP8 编码器和窗口")
        short, _ = infer_video(
            predictor.net,
            predictor.vae,
            source,
            predictor.device,
            max_frames=min(frames, 85),
            width=width,
            height=height,
            window_batch_size=16,
            resize_mode=resize_mode,
            long_edge=long_edge,
            geometry_intrinsics=geometry_intrinsics,
        )
        latent = torch.from_numpy(short["latent_mean"]).to(
            predictor.device, dtype=predictor.vae.dtype
        )
        needed = max(offset // 4 + length for _, _, offset, _, length, _ in missing)
        if latent.shape[1] < needed:
            latent = torch.cat(
                (latent, latent[:, -1:].expand(-1, needed - latent.shape[1], -1, -1)), 1
            )
        batches = []
        for _, _, offset, count, _, size in sorted(missing):
            batch = batch_for(
                offset,
                count,
                latent,
                predictor.device,
                height,
                width,
                geometry_intrinsics=geometry_intrinsics,
            )
            if size > 1:
                batch = combine_window_batches([batch] * size)
            batches.append(batch)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                predictor.net(batch)
        validate_deferred_camera(predictor.net)
        torch.cuda.synchronize(predictor.device)
    if width != 480:
        if state.qkv is None:
            state.qkv = CompiledQKV(gemm_amax=False)
        state.qkv.discover()
        if not state.qkv.patches:
            raise RuntimeError("No compiled QKV packing kernel was matched")
        state.qkv.configure(True)
    if missing:
        warm, _ = infer_video(
            predictor.net,
            predictor.vae,
            source,
            predictor.device,
            max_frames=min(frames, 85),
            width=width,
            height=height,
            window_batch_size=16,
            resize_mode=resize_mode,
            long_edge=long_edge,
            geometry_intrinsics=geometry_intrinsics,
        )
        del warm
        for batch in batches:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                predictor.net(batch)
        validate_deferred_camera(predictor.net)
        torch.cuda.synchronize(predictor.device)
        state.coverage.update(signatures)
        coverage.update(signatures)
    predictor.acceleration_evidence["exact_kernel_optimization"] = {"direct_pinned": False}
