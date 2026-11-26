import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

import numpy as np

from .mesh_raster import draw_mesh_roi
from .overlay_indicators import concentration_colors, draw_bars, indicator_label, prepare_indicators
from .overlay_pipeline import AsyncVideoWriter, prefetch_rgb
from .overlay_raster import PillowCanvas, RGBCanvas, draw_labels_roi, draw_skeleton_roi, label_font
from .video import VideoWriter

COLORS = ((30, 235, 140), (255, 80, 175))
EDGES = [(0, start) for start in (1, 5, 9, 13, 17)] + [
    (start + i, start + i + 1) for start in (1, 5, 9, 13, 17) for i in range(3)
]
PREDICTION_FIELDS = (
    "frame_indices",
    "existence_prob",
    "anchors_2d",
    "joints_3d",
    "vertices",
    "overlay_camera_valid",
    "overlay_intrinsics",
    "faces_left",
    "faces_right",
)


def load_overlay_prediction(path):
    with np.load(path, allow_pickle=False) as payload:
        result = {key: payload[key] for key in PREDICTION_FIELDS}
        scores = (
            payload["overlay_joint_concentration"]
            if "overlay_joint_concentration" in payload
            else None
        )
        if scores is not None:
            result["overlay_joint_concentration"] = scores
        return result


def project(points, k):
    points = np.asarray(points, dtype=np.float64)
    depth = points[..., 2]
    uv = points[..., :2] / np.maximum(depth[..., None], 1e-05)
    return (uv * np.array([k[0, 0], k[1, 1]]) + np.array([k[0, 2], k[1, 2]]), depth > 0.01)


def canvas_to_source(pixels, crop_affine):
    left, top, sx, sy = crop_affine
    return np.asarray(pixels) / [sx, sy] + [left, top]


def render(
    input_video,
    output_video,
    result,
    metadata,
    faces,
    threshold=0.5,
    mode="anchors",
    progress=None,
    *,
    pipeline_depth=2,
    encoder="auto",
    backend="auto",
    gpu_batch_size=2,
):
    if not isinstance(pipeline_depth, int) or not 0 <= pipeline_depth <= 8:
        raise ValueError("Overlay pipeline depth must be an integer from 0 to 8")
    if backend not in {"auto", "cpu", "cuda"}:
        raise ValueError("Unknown overlay backend")
    started = perf_counter()
    result = prepare_indicators(result, metadata)
    joint_colors = (
        concentration_colors(result["overlay_joint_concentration"])
        if "overlay_joint_concentration" in result
        else None
    )
    failed_gpu_seconds = 0.0
    fallback_reason = None
    width, height = metadata["source_size"]
    gpu_candidate = encoder != "libx264" and (
        backend == "cuda" or (width * height >= 1000000 and len(result["frame_indices"]) >= 240)
    )
    if backend == "cuda" and (not gpu_candidate):
        raise ValueError("CUDA overlay does not support custom backgrounds, alignment or libx264")
    if backend != "cpu" and gpu_candidate:
        from .gpu_overlay import GPUOverlayEncodingError, GPUOverlayUnavailable, render_cuda

        target = Path(output_video)
        with NamedTemporaryFile(
            prefix=f".{target.stem}.cuda-", suffix=target.suffix, dir=target.parent, delete=False
        ) as temporary:
            gpu_target = Path(temporary.name)
        attempt_started = perf_counter()
        try:
            count = render_cuda(
                input_video,
                gpu_target,
                result,
                metadata,
                faces,
                threshold,
                mode,
                progress,
                batch_size=gpu_batch_size,
            )
            gpu_target.replace(target)
            return count
        except (GPUOverlayUnavailable, GPUOverlayEncodingError) as error:
            if backend == "cuda":
                raise
            failed_gpu_seconds = perf_counter() - attempt_started
            fallback_reason = f"{type(error).__name__}: {error}"
            if isinstance(error, GPUOverlayEncodingError) and encoder == "auto":
                encoder = "libx264"
            logging.getLogger(__name__).warning(
                "CUDA overlay failed; restarting from source with CPU rendering: %s",
                fallback_reason,
            )
        finally:
            gpu_target.unlink(missing_ok=True)
    model_size = metadata["size"]
    width, height = metadata["source_size"]
    canvas_type = RGBCanvas if width * height >= 4000000 else PillowCanvas

    def to_canvas(uv):
        return canvas_to_source(uv, metadata["crop_affine"])

    writer = (
        AsyncVideoWriter(
            output_video, width, height, metadata["fps"], codec=encoder, depth=pipeline_depth
        )
        if pipeline_depth
        else VideoWriter(
            output_video, width, height, metadata["fps"], codec=encoder, faststart=True
        )
    )
    stats = {"decode_rgb_seconds": 0.0 if pipeline_depth else None, "draw_rgb_seconds": 0.0}
    prefetched = None
    if pipeline_depth:
        prefetched = prefetch_rgb(input_video, len(result["frame_indices"]), pipeline_depth, stats)
    font = label_font(width, height)
    rendered = 0
    try:
        for i, rgb in enumerate(prefetched):
            if i >= len(result["frame_indices"]):
                break
            if (rgb.shape[1], rgb.shape[0]) != (width, height):
                raise ValueError("Overlay background dimensions do not match requested canvas")
            draw_start = perf_counter()
            canvas = canvas_type(rgb)
            for hand in (0, 1):
                if result["existence_prob"][i, hand] < threshold:
                    continue
                color = COLORS[hand]
                if mode == "anchors":
                    uv = to_canvas(result["anchors_2d"][i, hand] * model_size)
                    valid = np.isfinite(uv).all(-1)
                else:
                    if not result["overlay_camera_valid"][i]:
                        continue
                    k = result["overlay_intrinsics"][i]
                    uv, valid = project(result["joints_3d"][i, hand], k)
                    uv = to_canvas(uv)
                    if mode == "mesh":
                        mesh_uv, mesh_valid = project(result["vertices"][i, hand], k)
                        mesh_uv = to_canvas(mesh_uv)
                        triangles = faces[hand]
                        depths = result["vertices"][i, hand, triangles, 2].mean(-1)
                        canvas = draw_mesh_roi(
                            canvas, mesh_uv, mesh_valid, depths, triangles, color
                        )
                valid &= np.isfinite(uv).all(-1) & (np.abs(uv).max(-1) < 10000)
                draw_skeleton_roi(
                    canvas,
                    uv,
                    valid,
                    EDGES,
                    color,
                    None if joint_colors is None else joint_colors[i, hand],
                )
            title = f"Reconstruction | {mode} | L green / R pink | Frame {int(result['frame_indices'][i])} | {metadata['timestamps'][i]:.3f}s"
            label = None
            if mode != "anchors":
                label = (
                    "camera: supplied"
                    if metadata["overlay_camera"] == "supplied"
                    else "camera: predicted pinhole"
                    if metadata["overlay_camera"] == "predicted_parametric_pinhole"
                    else "camera: estimated (approximate)"
                )
                if not result["overlay_camera_valid"][i]:
                    label += " / invalid; skipped"
            hint = indicator_label(result)
            label = hint if label is None else label + " | " + hint
            draw_labels_roi(canvas, title, label, font)
            draw_bars(canvas, result["existence_prob"][i], font)
            rgb = canvas.rgb.view()
            rgb.flags.writeable = False
            stats["draw_rgb_seconds"] += perf_counter() - draw_start
            writer.write(rgb, metadata["timestamps"][i])
            rendered += 1
            if progress and i % 30 == 0:
                progress(i + 1)
    finally:
        try:
            writer.close()
        finally:
            if prefetched is not None:
                prefetched.close()
    if rendered != len(result["frame_indices"]):
        raise ValueError("Overlay frame count mismatch")
    if progress:
        progress(rendered)
    metadata["overlay_render"] = {
        "backend": "cpu",
        "backend_fallback_reason": fallback_reason,
        "failed_gpu_seconds": failed_gpu_seconds,
        "encoder": writer.codec,
        "encoder_options": writer.options,
        "encoder_fallback_reason": writer.fallback_reason,
        "compression": "NVENC CQ18" if writer.codec == "h264_nvenc" else "libx264 CRF18 fast",
        "pipeline_depth": pipeline_depth,
        "pixel_format": "yuv420p",
        "raster_layout": "rgb_rois_global_bitmask" if canvas_type is RGBCanvas else "pillow_rgb",
        "frames": rendered,
        "seconds": perf_counter() - started,
        "stage_seconds_overlap": bool(pipeline_depth),
        **stats,
        **getattr(writer, "stats", {}),
    }
    return rendered
