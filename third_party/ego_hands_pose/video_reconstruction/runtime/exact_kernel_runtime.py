import json
import threading
from contextlib import contextmanager

import torch

from .runtime import ROOT

RECIPE = "fused"
EXACT_MANIFEST = ROOT / "assets/acceleration/exact.json"
NATIVE_MANIFEST = ROOT / "assets/acceleration/exact.json"
WIDE_MANIFEST = ROOT / "assets/acceleration/exact.json"
CANVAS832_MANIFEST = ROOT / "assets/acceleration/exact.json"
_LOCK = threading.RLock()


def recipe_choice(config):
    choice = config.get("exact_kernel_optimization", "off")
    if choice not in ("off", RECIPE):
        raise ValueError(f"Unknown exact kernel optimization: {choice}")
    configured_canvases(config)
    return choice


def configured_canvases(config):
    try:
        canvases = {tuple(size) for size in config.get("exact_kernel_canvases", [[480, 480]])}
    except TypeError as error:
        raise ValueError("Invalid exact_kernel_canvases") from error
    if not canvases <= {(480, 480), (672, 480), (832, 480)}:
        raise ValueError("Unvalidated exact_kernel_canvases")
    return canvases


def _validate(checkpoint, device):
    from .runtime import verify_assets

    manifest = json.loads(EXACT_MANIFEST.read_text())
    if (
        device.type != "cuda"
        or list(torch.cuda.get_device_capability(device)) != manifest["capability"]
        or str(torch.__version__) != manifest["torch"]
    ):
        raise ValueError("Fused FP8 kernels require their matching GPU and software environment")
    if checkpoint["weights_sha256"] != manifest["weights_sha256"]:
        raise ValueError("FP8 weights do not match the kernel configuration")
    verify_assets()
    return manifest


def validate_exact_assets(checkpoint, device):
    return [_validate(checkpoint, device)]


def validate_wide_assets(checkpoint, device):
    return _validate(checkpoint, device)


def validate_832_assets(checkpoint, device):
    return _validate(checkpoint, device)


class ExactKernelRuntime:
    def __init__(self):
        self.qkv = self.native = self.manual = None
        self.coverage = set()
        self.coverage_by_camera = {False: set(), True: set()}
        self.modules = None
        self.vae = None
        self.validated = set()
        self.manual_by_canvas = {}
        self.canvas832_manifest = None
        self.native_frozen = False

    def suspend(self):
        if self.native is not None:
            self.native.configure(False)
        if self.manual is not None:
            self.manual.configure(False)
        if self.qkv is not None:
            self.qkv.configure(False)

    def camera_coverage(self, geometry_intrinsics):
        return self.coverage_by_camera[geometry_intrinsics is not None]

    def prepare(
        self,
        predictor,
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
        choice = recipe_choice(predictor.config)
        evidence = dict(recipe=choice, enabled=False)
        predictor.acceleration_evidence["exact_kernel_optimization"] = evidence
        if choice == "off":
            evidence["reason"] = "disabled_in_config"
            return
        canvas = (width, height)
        is832 = canvas == (832, 480)
        allowed_mode = resize_mode == "center_crop" or (is832 and resize_mode == "long_edge_pad")
        if canvas not in configured_canvases(predictor.config) or not allowed_mode:
            evidence["reason"] = "canvas_not_validated; existing protected VAE recipe retained"
            return
        from .vae_kernel_runtime import RECIPE as VAE_RECIPE

        if predictor.config.get("vae_kernel_optimization") != VAE_RECIPE:
            raise ValueError(
                f"{RECIPE} requires {VAE_RECIPE}; set exact_kernel_optimization=off to use another VAE recipe"
            )
        if self.vae is not predictor.vae:
            self.suspend()
            self.__init__()
            self.vae = predictor.vae
        wide = canvas in {(672, 480), (832, 480)}
        if canvas not in self.validated:
            if is832:
                self.canvas832_manifest = validate_832_assets(
                    predictor.checkpoint, predictor.device
                )
            elif wide:
                validate_wide_assets(predictor.checkpoint, predictor.device)
            else:
                validate_exact_assets(predictor.checkpoint, predictor.device)
            self.validated.add(canvas)
        options = self.canvas832_manifest["options"] if is832 else {}
        use_qkv = not is832 or bool(options.get("qkv"))
        use_native = not is832 or bool(options.get("norm"))
        use_manual = wide and (not is832 or bool(options.get("manual")))
        direct_pinned = options.get("pinned", False) if is832 else not wide
        active_manifest = CANVAS832_MANIFEST if is832 else WIDE_MANIFEST if wide else EXACT_MANIFEST
        from torch._inductor.codecache import PyCodeCache

        from .accelerated_inference import (
            batch_for,
            combine_window_batches,
            infer_video,
            window_groups,
        )
        from .fp8_runtime import validate_deferred_camera
        from .models.components.sequence_fusion import latent_window, sequence_windows

        signatures = set()
        for group, size in window_groups(
            sequence_windows(frames, 81, 40), 1 + (frames + 2) // 4, 16
        ):
            selection, offset = latent_window(*group[0], 1 + (frames + 2) // 4)
            signatures.add(
                (width, height, offset, group[0][1], selection.stop - selection.start, size)
            )
        coverage = self.camera_coverage(geometry_intrinsics)
        missing = signatures - coverage
        if missing:
            progress(0.1, "预热当前视频的完整窗口形状和精确优化 kernel")
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
                direct_pinned=direct_pinned,
                geometry_intrinsics=geometry_intrinsics,
            )
            latent = torch.from_numpy(short["latent_mean"]).to(
                predictor.device, dtype=predictor.vae.dtype
            )
            needed = max((offset // 4 + length for _, _, offset, _, length, _ in missing))
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
                with torch.autocast(predictor.device.type, dtype=torch.bfloat16):
                    predictor.net(batch)
            validate_deferred_camera(predictor.net)
            torch.cuda.synchronize(predictor.device)
        from .kernel_compiled_qkv import CompiledQKV
        from .kernel_native_normpack import NativeNormpack
        from .kernel_wide_normpack import WideNormpack

        if self.qkv is None:
            self.qkv = CompiledQKV(gemm_amax=True)
        self.qkv.discover()
        if use_qkv and (not self.qkv.patches):
            raise RuntimeError("No warmed backbone shape matched the accepted exact kernels")
        modules = tuple((id(module) for module in PyCodeCache.modules))
        if self.native is None or self.modules != modules or (self.native_frozen != is832):
            previous = self.native
            previous_frozen = self.native_frozen
            if is832:
                from .kernel_832_native_normpack import FrozenNativeNormpack

                native_type = FrozenNativeNormpack
            else:
                native_type = NativeNormpack
            self.native = native_type(predictor.vae)
            if previous is not None and previous_frozen == is832:
                self.native.derived = previous.derived
                self.native.evidence = previous.evidence
                self.native.executed = previous.executed
            self.modules = modules
            self.native_frozen = is832
        if use_manual:
            if canvas not in self.manual_by_canvas:
                if is832:
                    from .kernel_832_normpack import Canvas832Normpack

                    patch = Canvas832Normpack(predictor.vae)
                else:
                    patch = WideNormpack(predictor.vae)
                self.manual_by_canvas[canvas] = patch
            self.manual = self.manual_by_canvas[canvas]
        self.qkv.configure(use_qkv)
        self.native.configure(use_native)
        if use_manual:
            self.manual.configure(True)
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
                direct_pinned=direct_pinned,
                geometry_intrinsics=geometry_intrinsics,
            )
            del warm
            for batch in batches:
                with torch.autocast(predictor.device.type, dtype=torch.bfloat16):
                    predictor.net(batch)
            validate_deferred_camera(predictor.net)
            torch.cuda.synchronize(predictor.device)
            self.coverage.update(signatures)
            coverage.update(signatures)
            if is832:
                del batches, batch, latent, short
                import gc

                gc.collect()
                torch.cuda.empty_cache()
        evidence.update(
            enabled=True,
            exact_kernels=True,
            native_normpack_compact=use_native,
            direct_pinned=direct_pinned,
            canvas_size=[width, height],
            wide_early_normpack=use_manual,
            exact_manifest=str(active_manifest),
            native_manifest=str(active_manifest if wide else NATIVE_MANIFEST),
            warmed_signatures=[list(item) for item in sorted(self.coverage) if item[:2] == canvas],
            warmup_reused=not bool(missing),
            backbone_shapes=len(self.qkv.patches),
            native_kernels=len(self.native.derived),
        )
        if use_manual:
            evidence["early_normpack"] = dict(
                self.manual.evidence, accepted=True, acceptance_manifest=str(active_manifest)
            )
        if is832:
            evidence["options"] = dict(options)
            evidence["native_normpack_scope"] = "deep"
            evidence["frozen_original_norm_launch"] = use_native
            evidence["resize_mode"] = resize_mode


@contextmanager
def exact_kernel_session(predictor):
    with _LOCK:
        runtime = getattr(predictor, "_exact_kernel_runtime", None)
        if runtime is None:
            runtime = predictor._exact_kernel_runtime = ExactKernelRuntime()
        runtime.suspend()
        try:
            yield runtime
        finally:
            runtime.suspend()


def release_exact_runtime(predictor):
    with _LOCK:
        runtime = getattr(predictor, "_exact_kernel_runtime", None)
        if runtime is not None:
            runtime.suspend()
            del predictor._exact_kernel_runtime
