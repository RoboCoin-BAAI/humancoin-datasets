from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F

from .fp8_evidence import e4m3_instructions


class NormSiLU(nn.Module):
    def __init__(self, norm):
        super().__init__()
        norm._compiled_call_impl = None
        self.norm = norm

    def forward(self, x):
        return F.silu(self.norm(x))


def fuse_vae(vae):
    torch._dynamo.config.recompile_limit = 256
    torch._dynamo.config.accumulated_recompile_limit = 1024
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanResidualBlock

    changed = []
    for name, child in list(vae.named_modules()):
        if isinstance(child, WanResidualBlock):
            for attribute in ("norm1", "norm2"):
                wrapped = NormSiLU(getattr(child, attribute))
                wrapped.compile(dynamic=False, fullgraph=True)
                setattr(child, attribute, wrapped)
                changed.append(name + "." + attribute)
            child.nonlinearity = nn.Identity()
    return changed


def decode_without_cpu_read(self, features, batch):
    from .camera import calibrated_camera_rays

    k = getattr(batch, "geometry_intrinsics", None)
    raw, fit = (
        self.camera_head(features)
        if k is None
        else calibrated_camera_rays(features, k, batch.image_size)
    )
    self._deferred_camera_checks.append(fit["valid"])
    return self._decode_outputs(raw, self.decoder(features, raw), batch, camera_fit=fit)


def residual_with_cache_views(self, x, feat_cache=None, feat_idx=None):
    feat_idx = [0] if feat_idx is None else feat_idx
    h = self.conv_shortcut(x)
    for norm, conv in ((self.norm1, self.conv1), (self.norm2, self.conv2)):
        x = self.nonlinearity(norm(x))
        if conv is self.conv2:
            x = self.dropout(x)
        if feat_cache is None:
            x = conv(x)
        else:
            idx = feat_idx[0]
            history = x[:, :, -2:]
            if history.shape[2] < 2 and feat_cache[idx] is not None:
                history = torch.cat((feat_cache[idx][:, :, -1:], history), 2)
            x = conv(x, feat_cache[idx])
            feat_cache[idx] = history
            feat_idx[0] += 1
    return x + h


def cache_views(vae):
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanResidualBlock

    names = []
    for name, child in vae.named_modules():
        if isinstance(child, WanResidualBlock):
            child.forward = MethodType(residual_with_cache_views, child)
            names.append(name)
    return names


PROTECTED_LAYERS = ("encoder.conv_in", "encoder.conv_out", "quant_conv")


def recipe_assets(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    assets = {}
    for name, expected in manifest["assets_sha256"].items():
        path = directory / name
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Protected FP8 recipe asset hash mismatch: {name}")
        assets[name] = json.loads(path.read_text())
    choices = assets["fp8_vae_choices.json"]
    for name in PROTECTED_LAYERS:
        if choices.get(name, {}).get("implementation") != "original":
            raise ValueError(f"Protected FP8 recipe must retain FP16: {name}")
    selected = [name for name, item in choices.items() if item["implementation"] == "implicit_fp8"]
    if len(selected) != 23:
        raise ValueError("Protected FP8 recipe requires exactly 23 selected VAE layers")
    if set(selected) - assets["fp8_vae_calibration.json"]["layers"].keys():
        raise ValueError("Missing FP8 VAE activation calibration")
    return (assets, manifest)


@torch.inference_mode()
def warm_encoder_and_windows(
    net,
    vae,
    source,
    device,
    width,
    height,
    *,
    resize_mode="center_crop",
    long_edge=832,
    geometry_intrinsics=None,
):
    from .accelerated_inference import batch_for, infer_video

    result, _ = infer_video(
        net,
        vae,
        source,
        device,
        max_frames=85,
        width=width,
        height=height,
        resize_mode=resize_mode,
        long_edge=long_edge,
        geometry_intrinsics=geometry_intrinsics,
    )
    latent = torch.from_numpy(result["latent_mean"]).to(device, dtype=torch.bfloat16)
    if latent.shape[1] < 22:
        latent = torch.cat((latent, latent[:, -1:].expand(-1, 22 - latent.shape[1], -1, -1)), 1)
    for first in (0, 2):
        with torch.autocast(device.type, dtype=torch.bfloat16):
            net(
                batch_for(
                    first,
                    81,
                    latent,
                    device,
                    height,
                    width,
                    geometry_intrinsics=geometry_intrinsics,
                )
            )
    torch.cuda.synchronize(device)
    return latent


def validate_deferred_camera(net):
    if net._deferred_camera_checks and (not bool(torch.cat(net._deferred_camera_checks).all())):
        raise ValueError("Nonfinite camera parameters during warmup")
    net._deferred_camera_checks = []


@torch.inference_mode()
def warm_protected_shapes(
    net,
    vae,
    source,
    device,
    *,
    width,
    height,
    frames,
    resize_mode="center_crop",
    long_edge=832,
    geometry_intrinsics=None,
):
    from .accelerated_inference import batch_for, combine_window_batches, window_groups
    from .fp8_modules import populate_rope_phases
    from .models.components.sequence_fusion import sequence_windows

    started = time.perf_counter()
    populate_rope_phases(net, [(height // 32, width // 32)])
    latent = warm_encoder_and_windows(
        net,
        vae,
        source,
        device,
        width,
        height,
        resize_mode=resize_mode,
        long_edge=long_edge,
        geometry_intrinsics=geometry_intrinsics,
    )
    latent = latent.to(dtype=vae.dtype)
    with torch.autocast(device.type, dtype=torch.bfloat16):
        for first in (0, 2):
            net(
                batch_for(
                    first,
                    81,
                    latent,
                    device,
                    height,
                    width,
                    geometry_intrinsics=geometry_intrinsics,
                )
            )
    sizes = {
        size
        for _, size in window_groups(sequence_windows(frames, 81, 40), 1 + (frames + 2) // 4, 16)
    }
    normal = batch_for(
        0, 81, latent, device, height, width, geometry_intrinsics=geometry_intrinsics
    )
    with torch.autocast(device.type, dtype=torch.bfloat16):
        for size in sorted(sizes):
            batch = combine_window_batches([normal] * size) if size > 1 else normal
            net(batch)
            net(batch)
    validate_deferred_camera(net)
    torch.cuda.synchronize(device)
    return dict(
        compile_search_warmup_seconds=time.perf_counter() - started,
        warmed_window_batch_sizes=sorted(sizes),
        warmed_latent_lengths=[21, 22],
        warm_signature=[width, height, frames, resize_mode, long_edge],
        runtime_canvas_size=[width, height],
        calibration_resolution=[480, 480],
        resolution_tuning="frozen_480_operator_choices_reused_with_actual_grid_prewarm",
    )


@torch.inference_mode()
def prepare_protected(
    net,
    vae,
    source,
    device,
    config,
    *,
    width,
    height,
    frames,
    resize_mode="center_crop",
    long_edge=832,
    geometry_intrinsics=None,
):
    from .acceleration import enable_compilation, prepare_profile
    from .compile_cache import CANVASES, PersistentCompileCache
    from .fp8_modules import ImplicitFP8Conv, install_fp8_vae, install_packing, install_rope
    from .runtime import ROOT

    if device.type != "cuda":
        raise ValueError("FP8 requires CUDA")
    cache = PersistentCompileCache(config, device)
    cache_evidence = cache.load()
    directory = config.get("acceleration_recipe_dir", ROOT / "assets/acceleration/fp8")
    assets, manifest = recipe_assets(directory)
    vae_file = ROOT / "weights/encoder.safetensors"
    with vae_file.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != manifest["vae_weights_sha256"]:
            raise ValueError("FP8 calibration does not match the pretrained VAE weights")
    choices = {
        name: dict(item, compute_dtype="float16")
        for name, item in assets["convolution_choices.json"].items()
    }
    vae.to(dtype=torch.float16)
    started = time.perf_counter()
    evidence = prepare_profile(net, vae, "fp8", choices=choices, compile_backbone=False)
    evidence["profile"] = "fp8"
    evidence["persistent_compile_cache"] = cache_evidence
    eager = time.perf_counter()
    warm_encoder_and_windows(
        net,
        vae,
        source,
        device,
        width,
        height,
        resize_mode=resize_mode,
        long_edge=long_edge,
        geometry_intrinsics=geometry_intrinsics,
    )
    torch.cuda.synchronize(device)
    evidence["eager_prewarm_seconds"] = time.perf_counter() - eager
    vae._acceleration_chunk_frames = 8
    evidence["vae_compiled_stateless_modules"] = enable_compilation(net, vae)
    evidence["compiled"] = True
    grids = {(h // 32, w // 32) for w, h in CANVASES} | {(height // 32, width // 32)}
    evidence["rope"] = install_rope(net, grids=sorted(grids), temporal_lengths=range(1, 23))
    evidence["norm_silu"] = fuse_vae(vae)
    evidence["cache_views"] = cache_views(vae)
    net._deferred_camera_checks = []
    net._decode = MethodType(decode_without_cpu_read, net)
    evidence["fused_packing"] = install_packing(vae, assets["pack_choices.json"])
    quant_started = time.perf_counter()
    evidence["fp8_vae_layers"] = install_fp8_vae(
        vae, assets["fp8_vae_choices.json"], assets["fp8_vae_calibration.json"]["layers"]
    )
    torch.cuda.synchronize(device)
    evidence["vae_quantization_prepare_seconds"] = time.perf_counter() - quant_started
    evidence["protected_fp16_layers"] = list(PROTECTED_LAYERS)
    for name in PROTECTED_LAYERS:
        child = vae.get_submodule(name)
        if isinstance(child, ImplicitFP8Conv) or child.compute_dtype != "float16":
            raise RuntimeError(f"Protected input/output precision changed: {name}")
    evidence.update(
        warm_protected_shapes(
            net,
            vae,
            source,
            device,
            width=width,
            height=height,
            frames=frames,
            resize_mode=resize_mode,
            long_edge=long_edge,
            geometry_intrinsics=geometry_intrinsics,
        )
    )
    evidence["cuda_graph"] = False
    evidence["recipe_manifest"] = manifest
    evidence["prepare_seconds"] = time.perf_counter() - started
    vae._persistent_compile_cache = cache
    evidence["fp8_vae_mma"] = {}
    for name, child in vae.named_modules():
        if isinstance(child, ImplicitFP8Conv):
            ptx = child.last_kernel.asm["ptx"]
            instructions = e4m3_instructions(ptx)
            if not instructions:
                raise RuntimeError(f"No executed E4M3 tensor-core kernel: {name}")
            evidence["fp8_vae_mma"][name] = instructions
    return evidence
