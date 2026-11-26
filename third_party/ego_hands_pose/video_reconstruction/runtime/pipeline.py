"""Camera-space reconstruction with one serialized resident model."""

from __future__ import annotations

import gc
import json
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import torch

from .camera import validate_intrinsics
from .canvas_padding import resize_plan
from .ffmpeg_video import count_video_frames, probe_video
from .runtime import (
    choose_device,
    digest,
    load_network,
    load_vae,
    read_config,
    select_precision,
)
from .vendor import transformer

_DEFAULT_ROPE = transformer.rope_apply_qk


def video_plan(video, resize_mode="center_crop"):
    size = probe_video(video)["source_size"]
    if size[0] <= size[1]:
        return resize_plan(size, "center_crop", width=480, height=480)
    return resize_plan(size, resize_mode, width=672, height=480, long_edge=832)


def save_arrays(path, arrays):
    import zipfile

    with zipfile.ZipFile(
        path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True
    ) as archive:
        for name, value in arrays.items():
            with archive.open(name + ".npy", "w", force_zip64=True) as stream:
                np.lib.format.write_array(stream, np.asarray(value), allow_pickle=False)


def read_arrays(path):
    with np.load(path, allow_pickle=False) as stored:
        return {name: stored[name] for name in stored.files}


class Predictor:
    def __init__(self, config=None, device=None, *, precision=None):
        self.config = read_config(config)
        if precision is not None:
            if precision not in ("auto", "bf16", "fp8"):
                raise ValueError("Unknown precision")
            self.config["precision"] = precision
        self.device = choose_device(device or self.config["device"])
        self.net = self.vae = None
        self.mode = None
        self.bf16_optimized = False
        self.lock = threading.RLock()
        self._warm = set()
        self._calibrated_warm = set()

    def _release(self):
        from .exact_kernel_runtime import release_exact_runtime

        release_exact_runtime(self)
        self.net = self.vae = None
        self.mode = None
        self.bf16_optimized = False
        self._warm.clear()
        self._calibrated_warm.clear()
        if hasattr(self, "acceleration_evidence"):
            del self.acceleration_evidence
        transformer.rope_apply_qk = _DEFAULT_ROPE
        from .gpu_overlay import clear_cuda_context_cache

        clear_cuda_context_cache()
        gc.collect()
        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()

    def unload(self):
        from .exact_kernel_runtime import exact_kernel_session

        with self.lock:
            with exact_kernel_session(self), torch.cuda.device(self.device):
                self._release()
            gc.collect()
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()

    def _load(self, mode, progress, *, bf16_optimized=False):
        if self.mode == mode and self.bf16_optimized == bf16_optimized and self.net is not None:
            return
        if self.net is not None:
            self._release()
        self.config["acceleration_profile"] = mode
        progress(0.02, "加载模型")
        self.net, self.checkpoint = load_network(self.config, self.device)
        progress(0.04, "加载视频编码器")
        self.vae = load_vae(self.config, self.device)
        self.mode = mode
        self.bf16_optimized = bf16_optimized

    def _prepare(self, source, plan, total, mode, state, progress, geometry_intrinsics=None):
        from .accelerated_inference import infer_video

        kw = dict(
            width=plan["size"][0],
            height=plan["size"][1],
            frames=total,
            resize_mode=plan["resize_mode"],
            long_edge=832,
        )
        signature = (mode, *plan["size"], plan["resize_mode"], total)
        warm_signatures = self._warm if geometry_intrinsics is None else self._calibrated_warm
        camera = dict(geometry_intrinsics=geometry_intrinsics)
        if mode == "fp8":
            from .fp8_runtime import prepare_protected, warm_protected_shapes
            from .vae_kernel_runtime import ensure_vae_kernel_recipe

            if not hasattr(self, "acceleration_evidence"):
                progress(0.06, "准备 FP8 加速与编译缓存")
                self.acceleration_evidence = prepare_protected(
                    self.net, self.vae, source, self.device, self.config, **kw, **camera
                )
                self._rope_operator = transformer.rope_apply_qk
            else:
                transformer.rope_apply_qk = self._rope_operator
                if signature not in warm_signatures:
                    progress(0.08, "预热当前输入形状")
                    self.acceleration_evidence.update(
                        warm_protected_shapes(
                            self.net, self.vae, source, self.device, **kw, **camera
                        )
                    )
            ensure_vae_kernel_recipe(self, source, progress=progress, **kw)
            if torch.cuda.get_device_capability(self.device) == (9, 0):
                from .hopper import prepare

                prepare(self, state, source, progress=progress, **kw, **camera)
            else:
                state.prepare(self, source, progress=progress, **kw, **camera)
        else:
            if not hasattr(self, "acceleration_evidence"):
                from .acceleration import prepare_profile
                from .compile_cache import PersistentCompileCache

                cache = PersistentCompileCache(self.config, self.device)
                cache.load()
                self.acceleration_evidence = prepare_profile(
                    self.net, self.vae, "bf16", compile_backbone=self.config["compile"]
                )
                if self.bf16_optimized:
                    from .bf16_runtime import prepare

                    self.acceleration_evidence["bf16_data_movement"] = prepare(self.net, self.vae)
                from .bf16_runtime import stabilize_convolutions, validated_device

                if validated_device(self.device):
                    self.acceleration_evidence["stable_convolutions"] = stabilize_convolutions(
                        self.net, self.vae
                    )
                self.vae._persistent_compile_cache = cache
            from .accelerated_inference import batch_for, combine_window_batches, window_groups
            from .models.components.sequence_fusion import latent_window, sequence_windows

            signatures = set()
            latent_frames = 1 + (total + 2) // 4
            for group, size in window_groups(
                sequence_windows(total, 81, 40),
                latent_frames,
                int(self.config["window_batch_size"]),
            ):
                selection, offset = latent_window(*group[0], latent_frames)
                signatures.add(
                    (
                        mode,
                        *plan["size"],
                        offset,
                        group[0][1],
                        selection.stop - selection.start,
                        size,
                    )
                )
            missing = signatures - warm_signatures
            if missing:
                progress(0.08, "预热 BF16 推理")
                warm, _ = infer_video(
                    self.net,
                    self.vae,
                    source,
                    self.device,
                    max_frames=min(total, 85),
                    width=kw["width"],
                    height=kw["height"],
                    resize_mode=kw["resize_mode"],
                    window_batch_size=int(self.config["window_batch_size"]),
                    **camera,
                )
                latent = torch.from_numpy(warm["latent_mean"]).to(self.device, dtype=self.vae.dtype)
                needed = max(offset // 4 + length for _, _, _, offset, _, length, _ in missing)
                if latent.shape[1] < needed:
                    latent = torch.cat(
                        (latent, latent[:, -1:].expand(-1, needed - latent.shape[1], -1, -1)), 1
                    )
                for _, _, _, offset, count, _, size in sorted(missing):
                    batch = batch_for(
                        offset, count, latent, self.device, kw["height"], kw["width"], **camera
                    )
                    if size > 1:
                        batch = combine_window_batches([batch] * size)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        self.net(batch)
                warm_signatures.update(signatures)
        warm_signatures.add(signature)
        torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def run(
        self,
        video,
        output_dir,
        *,
        resize_mode="center_crop",
        max_frames=None,
        overlay_mode="joints",
        threshold=0.5,
        intrinsics=None,
        geometry_intrinsics=None,
        global_correction=False,
        render_overlay=True,
        width=None,
        height=None,
        progress=None,
    ):
        """Reconstruct using optional fixed K in original-video pixel coordinates."""
        geometry_intrinsics = validate_intrinsics(geometry_intrinsics)
        from .exact_kernel_runtime import exact_kernel_session
        from .hopper import execution_policy

        progress = progress or (lambda value, text: print(text, flush=True))
        with (
            self.lock,
            exact_kernel_session(self) as state,
            torch.cuda.device(self.device),
            execution_policy(self.device),
        ):
            try:
                return self._run(
                    video,
                    output_dir,
                    resize_mode,
                    max_frames,
                    overlay_mode,
                    threshold,
                    intrinsics,
                    geometry_intrinsics,
                    global_correction,
                    render_overlay,
                    width,
                    height,
                    progress,
                    state,
                )
            except Exception:
                self._release()
                raise
            finally:
                transformer.rope_apply_qk = _DEFAULT_ROPE

    def _run(
        self,
        video,
        output_dir,
        resize_mode,
        max_frames,
        overlay_mode,
        threshold,
        intrinsics,
        geometry_intrinsics,
        global_correction,
        render_overlay,
        width,
        height,
        progress,
        state,
    ):
        import av

        from .accelerated_inference import FIELDS, infer_video
        from .compile_cache import persist_request_cache
        from .overlay_indicators import joint_concentration

        video = Path(video).resolve(strict=True)
        output = Path(output_dir).resolve()
        if output.exists():
            raise FileExistsError("Output directory already exists")
        limit = int(self.config["max_frames"] if max_frames is None else max_frames)
        if (
            limit < 0
            or not 0 <= threshold <= 1
            or overlay_mode not in ("anchors", "joints", "mesh")
        ):
            raise ValueError("Invalid frame limit, threshold or overlay mode")
        k = validate_intrinsics(intrinsics)
        if global_correction and k is not None:
            raise ValueError(
                "Clear the visualization intrinsics before enabling global consistency"
            )
        plan = video_plan(video, resize_mode)
        if width is not None or height is not None:
            if width is None or height is None:
                raise ValueError("Width and height must be supplied together")
            plan = resize_plan(
                probe_video(video)["source_size"],
                resize_mode,
                width=int(width),
                height=int(height),
                long_edge=832,
            )
        geometry_canvas = (
            validate_intrinsics(np.asarray(plan["source_to_canvas"]) @ geometry_intrinsics)
            if geometry_intrinsics is not None
            else None
        )
        if self.config["compile"]:
            for modern, legacy, value in (
                ("recompile_limit", "cache_size_limit", 256),
                ("accumulated_recompile_limit", "accumulated_cache_size_limit", 1024),
            ):
                name = modern if hasattr(torch._dynamo.config, modern) else legacy
                setattr(torch._dynamo.config, name, max(getattr(torch._dynamo.config, name), value))
        mode = select_precision(
            self.config["precision"],
            torch.cuda.get_device_capability(self.device),
            str(torch.__version__),
            plan["size"],
            plan["resize_mode"],
            compile_enabled=self.config["compile"],
        )
        started = time.perf_counter()
        with av.open(str(video)) as container:
            capacity = container.streams.video[0].frames
        if not capacity:
            progress(0.005, "统计视频帧数")
            capacity = count_video_frames(video)
        total = min(capacity, limit) if limit else capacity
        if total <= 0:
            raise ValueError("No video frames")
        counted = time.perf_counter()
        bf16_optimized = False
        if mode == "bf16":
            from .bf16_runtime import supported

            bf16_optimized = supported(self.device, plan["size"], self.config["compile"])
        self._load(mode, progress, bf16_optimized=bf16_optimized)
        loaded = time.perf_counter()
        self._prepare(video, plan, total, mode, state, progress, geometry_canvas)
        prepared = time.perf_counter()

        def track(stage, count):
            text = {
                "vae": "视频编码",
                "window": "窗口推理",
                "fusion": "融合结果",
                "transfer": "传输结果",
            }[stage]
            fraction = count / total if stage == "vae" else 0.5 if stage == "window" else 0.9
            progress(0.12 + 0.5 * min(1, fraction), f"{text} {count}")

        arrays, metadata = infer_video(
            self.net,
            self.vae,
            video,
            self.device,
            max_frames=limit,
            expected_frames=capacity,
            return_latent=False,
            width=plan["size"][0],
            height=plan["size"][1],
            window_batch_size=16 if mode == "fp8" else int(self.config["window_batch_size"]),
            resize_mode=plan["resize_mode"],
            long_edge=832,
            progress=track,
            direct_pinned=bool(
                self.acceleration_evidence.get("exact_kernel_optimization", {}).get("direct_pinned")
            ),
            geometry_intrinsics=geometry_canvas,
        )
        scores = joint_concentration(arrays)
        keep = set(FIELDS) | {
            "frame_indices",
            "timestamps",
            "predicted_intrinsics",
            "overlay_intrinsics",
            "predicted_camera_valid",
            "overlay_camera_valid",
        }
        arrays = {name: value for name, value in arrays.items() if name in keep}
        if scores is not None:
            arrays["overlay_joint_concentration"] = scores
        arrays.update(faces_left=self.net.mano.left.faces, faces_right=self.net.mano.right.faces)
        cache = persist_request_cache(self)
        keys = (
            "frames",
            "size",
            "source_size",
            "fps",
            "timestamps",
            "crop_affine",
            "source_to_canvas",
            "resize_mode",
            "padding",
            "scaled_size",
            "intermediate_video",
            "truncated",
            "timings",
            "window",
            "stride",
            "windows",
            "window_batch_size",
            "vae_chunk_frames",
            "peak_allocated_bytes",
        )
        metadata = {key: metadata[key] for key in keys if key in metadata}
        metadata.update(
            input=video.name,
            source_sha256=digest(video),
            source_frames=capacity,
            device=str(self.device),
            precision=mode,
            architecture="parametric_pinhole",
            overlay_camera="predicted_parametric_pinhole",
            postprocessing="off",
            weights_sha256=self.checkpoint["weights_sha256"],
            overlay_generated=False,
            source_coordinates="OpenCV: X right, Y down, Z forward; metres",
            threshold=threshold,
            overlay_mode=overlay_mode,
            intermediate_video=False,
            visualization_intrinsics=k.tolist() if k is not None else None,
            geometry_camera_source="calibration"
            if geometry_intrinsics is not None
            else "predicted",
            geometry_intrinsics_source=(
                geometry_intrinsics.tolist() if geometry_intrinsics is not None else None
            ),
            geometry_intrinsics_canvas=geometry_canvas.tolist()
            if geometry_canvas is not None
            else None,
            decoder_and_pnp_use_calibration=geometry_intrinsics is not None,
        )
        if geometry_intrinsics is not None:
            metadata["overlay_camera"] = "calibrated_parametric_pinhole"
        metadata["timings"].update(
            frame_count_seconds=counted - started,
            load_seconds=loaded - counted,
            prepare_warmup_seconds=prepared - loaded,
            compile_cache_save_seconds=cache["seconds"],
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".pending-", dir=output.parent) as folder:
            pending = Path(folder)
            progress(0.7, "保存预测参数")
            save_arrays(pending / "prediction.npz", arrays)
            metadata["prediction_sha256"] = digest(pending / "prediction.npz")
            (pending / "metadata.json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
            )
            selected = self._select(pending, global_correction, progress)
            selected_meta = json.loads((selected / "metadata.json").read_text())
            if render_overlay:
                self._render(video, selected, selected_meta, overlay_mode, threshold, k, progress)
            selected_meta["timings"]["total_seconds"] = time.perf_counter() - started
            (selected / "metadata.json").write_text(
                json.dumps(selected_meta, indent=2, ensure_ascii=False) + "\n"
            )
            relative = selected.relative_to(pending)
            pending.rename(output)
        progress(1, "处理完成")
        return self._result(output / relative)

    def _select(self, root, enabled, progress):
        if not enabled:
            return root
        selected = root / "consistent"
        if (selected / "prediction.npz").exists():
            cached = json.loads((selected / "metadata.json").read_text())
            if cached.get("raw_prediction_sha256") != digest(root / "prediction.npz") or cached[
                "prediction_sha256"
            ] != digest(selected / "prediction.npz"):
                raise ValueError("Corrected prediction checksum mismatch")
            return selected
        from .global_consistency import apply_global_consistency

        progress(0.72, "统一整段视频的手形和相机")
        metadata = json.loads((root / "metadata.json").read_text())
        arrays = read_arrays(root / "prediction.npz")
        mano = self.net.mano if self.net is not None else None
        if mano is None:
            from .models.components.mano import BatchedMANO

            mano = BatchedMANO(self.config["mano_dir"]).to(self.device).eval()
        started = time.perf_counter()
        fixed_k = (
            metadata["geometry_intrinsics_source"]
            if metadata.get("geometry_camera_source") == "calibration"
            else None
        )
        updates, report = apply_global_consistency(
            arrays, metadata, mano, device=self.device, fixed_intrinsics_source=fixed_k
        )
        arrays.update(updates)
        selected.mkdir(exist_ok=False)
        save_arrays(selected / "prediction.npz", arrays)
        metadata.update(
            raw_prediction_sha256=metadata["prediction_sha256"],
            postprocessing="global_consistency",
            overlay_camera=(
                "calibrated_parametric_pinhole"
                if fixed_k is not None
                else "global_postprocessed_parametric_pinhole"
            ),
            global_consistency=report,
            prediction_sha256=digest(selected / "prediction.npz"),
            overlay_generated=False,
        )
        metadata["timings"]["global_consistency_seconds"] = time.perf_counter() - started
        (selected / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
        )
        return selected

    def _render(self, video, folder, metadata, mode, threshold, k, progress):
        from .overlay import load_overlay_prediction, render

        arrays = load_overlay_prediction(folder / "prediction.npz")
        display_meta = dict(metadata, overlay_mode=mode, threshold=threshold)
        if k is not None:
            left, top, sx, sy = metadata["crop_affine"]
            transform = np.array([[sx, 0, -sx * left], [0, sy, -sy * top], [0, 0, 1.0]])
            arrays["overlay_intrinsics"] = np.broadcast_to(
                transform @ k, arrays["overlay_intrinsics"].shape
            ).copy()
            arrays["overlay_camera_valid"] = np.ones(metadata["frames"], dtype=bool)
            display_meta["overlay_camera"] = "supplied"
        start = time.perf_counter()
        progress(0.8, "绘制叠加视频")
        backend = self.config["overlay_backend"]
        encoder = self.config["overlay_encoder"]
        with tempfile.NamedTemporaryFile(
            dir=folder, suffix=".mp4", prefix=".overlay-", delete=False
        ) as stream:
            temporary = Path(stream.name)
        try:
            render(
                video,
                temporary,
                arrays,
                display_meta,
                [arrays["faces_left"], arrays["faces_right"]],
                threshold,
                mode,
                backend=backend,
                encoder=encoder,
                progress=lambda n: progress(
                    0.8 + 0.19 * n / metadata["frames"], f"绘制 {n}/{metadata['frames']}"
                ),
            )
            temporary.replace(folder / "overlay.mp4")
        finally:
            temporary.unlink(missing_ok=True)
        metadata.update(
            overlay_render=display_meta["overlay_render"],
            overlay_generated=True,
            overlay_mode=mode,
            threshold=threshold,
            visualization_intrinsics=k.tolist() if k is not None else None,
        )
        metadata["timings"]["render_seconds"] = time.perf_counter() - start

    @torch.inference_mode()
    def rerender(
        self,
        video,
        result_dir,
        *,
        overlay_mode="joints",
        threshold=0.5,
        intrinsics=None,
        global_correction=False,
        progress=None,
    ):
        from .exact_kernel_runtime import exact_kernel_session

        progress = progress or (lambda value, text: None)
        k = validate_intrinsics(intrinsics)
        if not 0 <= threshold <= 1 or overlay_mode not in ("anchors", "joints", "mesh"):
            raise ValueError("Invalid render settings")
        if global_correction and k is not None:
            raise ValueError(
                "Clear the visualization intrinsics before enabling global consistency"
            )
        root = Path(result_dir).resolve()
        metadata = json.loads((root / "metadata.json").read_text())
        if digest(root / "prediction.npz") != metadata["prediction_sha256"]:
            raise ValueError("Prediction checksum mismatch")
        if digest(video) != metadata["source_sha256"]:
            raise ValueError("The source video has changed; run inference again")
        with self.lock, exact_kernel_session(self), torch.cuda.device(self.device):
            selected = self._select(root, global_correction, progress)
            metadata = json.loads((selected / "metadata.json").read_text())
            self._render(video, selected, metadata, overlay_mode, threshold, k, progress)
            (selected / "metadata.json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
            )
        return self._result(selected)

    @staticmethod
    def _result(folder):
        return dict(
            prediction=str(folder / "prediction.npz"),
            metadata=str(folder / "metadata.json"),
            overlay=str(folder / "overlay.mp4") if (folder / "overlay.mp4").exists() else None,
            summary=json.loads((folder / "metadata.json").read_text()),
            raw_directory=str(folder.parent if folder.name == "consistent" else folder),
        )
