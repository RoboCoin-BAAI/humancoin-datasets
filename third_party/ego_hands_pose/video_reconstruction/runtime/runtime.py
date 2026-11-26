"""Local asset loading and hardware selection."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
_VERIFIED = None


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_config(path=None):
    config = dict(
        device="cuda:0",
        precision="auto",
        cpu_threads=4,
        window_batch_size=1,
        compile=True,
        overlay_backend="auto",
        overlay_encoder="auto",
        max_frames=0,
    )
    if path is not None:
        config.update(json.loads(Path(path).read_text()))
    if config["precision"] not in ("auto", "bf16", "fp8"):
        raise ValueError("Precision must be auto, bf16 or fp8")
    if int(config["window_batch_size"]) < 1 or int(config["max_frames"]) < 0:
        raise ValueError("Invalid batch size or frame limit")
    config.update(
        mano_dir=str(ROOT / "assets/geometry"),
        acceleration_recipe_dir=str(ROOT / "assets/acceleration/fp8"),
        vae_kernel_optimization="fused",
        exact_kernel_optimization="fused",
        exact_kernel_canvases=[[480, 480], [672, 480], [832, 480]],
    )
    return config


def choose_device(request="cuda:0"):
    if request == "auto":
        request = "cuda:0"
    device = torch.device(request)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError("CUDA device index is out of range")
    return torch.device("cuda", index)


def verify_assets():
    global _VERIFIED
    manifest = json.loads((ROOT / "assets/manifest.json").read_text())
    identity = tuple(
        (
            (name, (ROOT / name).stat().st_size, (ROOT / name).stat().st_mtime_ns)
            for name in manifest["sha256"]
        )
    )
    if identity != _VERIFIED:
        for name, expected in manifest["sha256"].items():
            if digest(ROOT / name) != expected:
                raise ValueError(f"Asset checksum mismatch: {name}")
        _VERIFIED = identity
    return manifest


def select_precision(
    requested, capability, torch_version, canvas, resize_mode, *, compile_enabled=True
):
    """Pure hardware routing; unsupported kernels are never imported or installed."""
    supported_canvas = tuple(canvas) in {(480, 480), (672, 480), (832, 480)}
    crop = resize_mode == "center_crop"
    ada = (
        tuple(capability) == (8, 9)
        and torch_version == "2.13.0+cu130"
        and (crop or (tuple(canvas) == (832, 480) and resize_mode == "long_edge_pad"))
    )
    hopper = (
        tuple(capability) == (9, 0)
        and torch_version in {"2.13.0a0+8145d630e8.nv26.06", "2.13.0+cu130"}
        and crop
    )
    supported = supported_canvas and compile_enabled and (ada or hopper)
    if requested == "fp8" and (not supported):
        raise ValueError(
            "FP8 requires a supported GPU, software environment and canvas; use auto or bf16"
        )
    return "fp8" if requested != "bf16" and supported else "bf16"


def load_network(config, device):
    from .models.components.backbone import FeatureBackbone
    from .models.components.mano import BatchedMANO
    from .models.network import ReconstructionNetwork

    verify_assets()
    torch.set_num_threads(int(config["cpu_threads"]))
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    torch.set_float32_matmul_precision("highest")
    context = load_file(str(ROOT / "assets/context.safetensors"))["context"]
    if context.ndim == 2:
        context = context[None]
    backbone = FeatureBackbone(json.loads((ROOT / "assets/network.json").read_text()), context)
    net = ReconstructionNetwork(backbone, BatchedMANO(config["mano_dir"]))
    state = load_file(str(ROOT / "weights/model.safetensors"))
    net.load_state_dict(state, strict=True, assign=True)
    del state
    net.requires_grad_(False).eval().to(device)
    net.backbone.wan.freqs = net.backbone.wan.freqs.to(device)
    net.backbone.prepare_conditioning()
    return (
        net,
        {
            "weights_sha256": json.loads((ROOT / "assets/manifest.json").read_text())["sha256"][
                "weights/model.safetensors"
            ]
        },
    )


def load_vae(config, device):
    from diffusers import AutoencoderKLWan

    with torch.device("meta"):
        vae = AutoencoderKLWan.from_config(json.loads((ROOT / "assets/encoder.json").read_text()))
    state = load_file(str(ROOT / "weights/encoder.safetensors"))
    required = {name for name in vae.state_dict() if name.startswith(("encoder.", "quant_conv."))}
    if set(state) != required:
        raise ValueError("Encoder tensors do not match the architecture")
    dtype = torch.float16 if config["acceleration_profile"] == "fp8" else torch.bfloat16
    state = {name: value.to(torch.bfloat16).to(dtype) for name, value in state.items()}
    vae.load_state_dict(state, strict=False, assign=True)
    vae.decoder = vae.post_quant_conv = torch.nn.Identity()
    vae._cached_conv_counts["decoder"] = 0
    vae.clear_cache()
    return vae.requires_grad_(False).eval().to(device)
