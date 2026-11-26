import hashlib
import json
import time
from pathlib import Path

import torch

from .fp8_evidence import e4m3_instructions
from .runtime import ROOT

RECIPE = "fused"
DIRECTORY = ROOT / "assets/acceleration/iterator"
LAYOUT_DIRECTORY = ROOT / "assets/acceleration/layout"
NORMPACK_DIRECTORY = ROOT / "assets/acceleration/normpack"
PROTECTED = ("encoder.conv_in", "encoder.conv_out", "quant_conv")


def read_kernel_recipe(directory=DIRECTORY):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    content = (directory / "choices.json").read_bytes()
    if hashlib.sha256(content).hexdigest() != manifest["choices_sha256"]:
        raise ValueError("VAE kernel recipe choices hash mismatch")
    if manifest.get("protected_layers") != list(PROTECTED):
        raise ValueError("VAE kernel recipe changed the protected layers")
    choices = json.loads(content)
    for name, choice in choices.items():
        if name in PROTECTED or not name.startswith("encoder."):
            raise ValueError(f"VAE kernel recipe must only change internal encoder layers: {name}")
        if choice["mode"] not in {"original", "direct", "incremental"}:
            raise ValueError(f"Unmeasured VAE kernel mode: {name}")
    return (choices, manifest)


@torch.inference_mode()
def read_layout_recipe(directory=LAYOUT_DIRECTORY):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    content = (directory / "choices.json").read_bytes()
    if hashlib.sha256(content).hexdigest() != manifest["choices_sha256"]:
        raise ValueError("VAE weight layout choices hash mismatch")
    choices = json.loads(content)
    frozen, _ = read_kernel_recipe()
    for name, choice in choices.items():
        if name not in frozen or frozen[name]["mode"] == "original" or name in PROTECTED:
            raise ValueError(f"Unmeasured VAE weight layout layer: {name}")
        if set(choice) - {"stages", "maxnreg"} or choice.get("stages") not in (2, 3):
            raise ValueError(f"Unmeasured VAE weight layout parameters: {name}")
        if "maxnreg" in choice and choice["maxnreg"] != 128:
            raise ValueError(f"Unmeasured VAE weight layout register cap: {name}")
    return (choices, manifest)


def read_normpack_recipe(directory=NORMPACK_DIRECTORY):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    content = (directory / "choices.json").read_bytes()
    if hashlib.sha256(content).hexdigest() != manifest["choices_sha256"]:
        raise ValueError("VAE normpack choices hash mismatch")
    choices = json.loads(content)
    measured = {f"encoder.down_blocks.0.resnets.{r}.conv{c}" for r in (0, 1) for c in (1, 2)}
    if set(choices) != measured or any((c != {"rows": 4, "warps": 4} for c in choices.values())):
        raise ValueError("Unmeasured VAE normpack configuration")
    if manifest["scope"] != {
        "batch": 1,
        "channels": 160,
        "times": [1, 8],
        "spatial": [240, 240],
        "channel_stride": 1,
        "dtype": "float16",
    }:
        raise ValueError("Unmeasured VAE normpack scope")
    return (choices, manifest)


@torch.inference_mode()
def install_vae_kernel_recipe(vae):
    from .fp8_modules import ImplicitFP8Conv
    from .kernel_vae_kernels import fuse_avg_shortcuts, install_iterators, remove_downblock_clones
    from .kernel_vae_residual import fuse_residual_epilogues
    from .kernel_vae_weight_layout import install_weight_layout

    if hasattr(vae, "_vae_kernel_recipe"):
        if vae._vae_kernel_recipe["recipe"] != RECIPE:
            raise ValueError("Changing an installed VAE recipe requires reloading the model")
        return vae._vae_kernel_recipe
    choices, manifest = read_kernel_recipe()
    protected = {name: vae.get_submodule(name) for name in PROTECTED}
    for name, child in protected.items():
        if isinstance(child, ImplicitFP8Conv) or child.compute_dtype != "float16":
            raise ValueError(f"VAE kernel recipe requires protected FP16: {name}")
    frozen = {}
    for name, choice in choices.items():
        child = vae.get_submodule(name)
        if not isinstance(child, ImplicitFP8Conv):
            raise ValueError(f"VAE kernel recipe requires an existing calibrated FP8 layer: {name}")
        if choice["mode"] != "original":
            frozen[name] = (child.fp8_weight, child.activation_scale, child.weight_scale)
    evidence = dict(
        recipe=RECIPE,
        iterator_layers=install_iterators(vae, choices),
        removed_downblock_clones=remove_downblock_clones(vae),
        fused_avg_shortcuts=fuse_avg_shortcuts(vae),
        fused_residual_layers=fuse_residual_epilogues(vae),
        protected_fp16_layers=list(PROTECTED),
        precision="original FP8 scales/weights and intermediate FP16 rounding retained",
        choices_sha256=manifest["choices_sha256"],
        tuning_canvas=[480, 480],
        warmed_canvases=[],
        source_sha256={
            name: hashlib.sha256((ROOT / "runtime" / name).read_bytes()).hexdigest()
            for name in ("kernel_vae_kernels.py", "kernel_vae_residual.py", "vae_kernel_runtime.py")
        },
    )
    for name, previous in frozen.items():
        child = vae.get_submodule(name)
        actual = (child.fp8_weight, child.activation_scale, child.weight_scale)
        if not all((a is b for a, b in zip(actual, previous, strict=True))):
            raise RuntimeError(f"VAE kernel changed calibrated weights/scales: {name}")
    layout_choices, layout_manifest = read_layout_recipe()
    evidence["weight_layout_layers"] = install_weight_layout(vae, layout_choices)
    evidence["weight_layout_choices_sha256"] = layout_manifest["choices_sha256"]
    evidence["source_sha256"]["kernel_vae_weight_layout.py"] = hashlib.sha256(
        (ROOT / "runtime/kernel_vae_weight_layout.py").read_bytes()
    ).hexdigest()
    for name in layout_choices:
        child = vae.get_submodule(name)
        weight, activation_scale, weight_scale = frozen[name]
        restored = child.fp8_weight.t().reshape(weight.shape)
        if not torch.equal(restored.view(torch.uint8), weight.view(torch.uint8)):
            raise RuntimeError(f"VAE layout changed calibrated FP8 bytes: {name}")
        if child.activation_scale is not activation_scale or child.weight_scale is not weight_scale:
            raise RuntimeError(f"VAE layout changed calibrated scales: {name}")
    from .kernel_vae_normpack import install_normpack

    normpack_choices, normpack_manifest = read_normpack_recipe()
    evidence["normpack_layers"] = install_normpack(vae, normpack_choices)
    evidence["normpack_choices_sha256"] = normpack_manifest["choices_sha256"]
    evidence["normpack_scope"] = normpack_manifest["scope"]
    evidence["source_sha256"]["kernel_vae_normpack.py"] = hashlib.sha256(
        (ROOT / "runtime/kernel_vae_normpack.py").read_bytes()
    ).hexdigest()
    if not all((vae.get_submodule(name) is child for name, child in protected.items())):
        raise RuntimeError("VAE kernel replaced a protected layer")
    vae.clear_cache()
    vae._vae_kernel_recipe = evidence
    return evidence


def verify_vae_kernel_execution(vae, evidence, width, height):
    instructions = {}
    for name in evidence["iterator_layers"]:
        child = vae.get_submodule(name)
        kernel = getattr(child, "last_kernel", None)
        ptx = kernel.asm["ptx"] if kernel is not None else ""
        mma = e4m3_instructions(ptx)
        if not mma:
            raise RuntimeError(f"VAE optimized E4M3 kernel did not execute: {name}")
        instructions[name] = mma
    evidence["executed_fp8_mma"] = instructions
    if evidence.get("normpack_layers") and (width, height) == (480, 480):
        conversions = {}
        for name in evidence["normpack_layers"]:
            kernel = getattr(vae.get_submodule(name), "last_normpack_kernel", None)
            ptx = kernel.asm["ptx"] if kernel is not None else ""
            lines = sorted(
                {line.strip() for line in ptx.splitlines() if "cvt" in line and "e4m3" in line}
            )
            if not lines:
                raise RuntimeError(f"VAE normpack FP8 conversion did not execute: {name}")
            conversions[name] = lines
        evidence["executed_normpack_fp8_conversion"] = conversions
    canvas = [width, height]
    if canvas not in evidence["warmed_canvases"]:
        evidence["warmed_canvases"].append(canvas)


@torch.inference_mode()
def ensure_vae_kernel_recipe(
    predictor, source, *, width, height, frames, resize_mode, long_edge, progress
):
    from .accelerated_inference import encode_double_buffer
    from .canvas_padding import resize_plan
    from .ffmpeg_video import ffmpeg_frames, probe_video

    started = time.perf_counter()
    evidence = install_vae_kernel_recipe(predictor.vae)
    if [width, height] not in evidence["warmed_canvases"]:
        progress(0.1, "预热 VAE 卷积迭代器和残差融合 kernel")
        plan = resize_plan(
            probe_video(source)["source_size"],
            resize_mode,
            width=width,
            height=height,
            long_edge=long_edge,
        )
        count = min(frames, 85)
        latent, _ = encode_double_buffer(
            predictor.vae, ffmpeg_frames(source, plan, {}, count), predictor.device, count
        )
        torch.cuda.synchronize(predictor.device)
        verify_vae_kernel_execution(predictor.vae, evidence, width, height)
        del latent
    evidence["last_prepare_seconds"] = time.perf_counter() - started
    predictor.acceleration_evidence["vae_kernel_optimization"] = evidence
