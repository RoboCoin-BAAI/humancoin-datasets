from __future__ import annotations

import queue
import threading
import time
from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import torch

from .camera import validate_intrinsics
from .canvas_padding import resize_plan
from .ffmpeg_video import ffmpeg_frames, probe_video
from .models.components.sequence_fusion import SequenceFusion, latent_window, sequence_windows

FIELDS = (
    "global_orient",
    "hand_pose",
    "betas",
    "translation",
    "joints_3d",
    "vertices",
    "anchors_2d",
    "existence_prob",
    "visibility_prob",
)


@torch.inference_mode()
def encode_double_buffer(vae, frames, device, capacity, *, progress=None):
    from diffusers.models.autoencoders.autoencoder_kl_wan import patchify

    chunk_frames = getattr(vae, "_acceleration_chunk_frames", 4)
    if chunk_frames < 4 or chunk_frames % 4:
        raise ValueError("VAE chunk size must be a positive multiple of four")
    iterator = iter(frames)
    first_frame = next(iterator)
    h, w = first_frame.shape[:2]
    host = [
        torch.empty((chunk_frames, h, w, 3), dtype=torch.uint8, pin_memory=True) for _ in range(2)
    ]
    gpu = [torch.empty_like(host[0], device=device) for _ in range(2)]
    events = [torch.cuda.Event() for _ in range(2)]
    occupied = [False, False]
    free, ready = (queue.Queue(), queue.Queue(maxsize=2))
    for slot in range(2):
        free.put(slot)
    stop = threading.Event()
    errors = []

    @torch.inference_mode()
    def producer():
        try:
            first = True
            while not stop.is_set():
                try:
                    slot = free.get(timeout=0.1)
                except queue.Empty:
                    continue
                if occupied[slot]:
                    events[slot].synchronize()
                count = 0
                size = 1 if first else chunk_frames
                for index in range(size):
                    try:
                        frame = first_frame if first else next(iterator)
                    except StopIteration:
                        break
                    host[slot][index].copy_(torch.from_numpy(np.array(frame, copy=True)))
                    count += 1
                first = False
                if count == 0:
                    ready.put(None)
                    return
                for index in range(count, size):
                    host[slot][index].copy_(host[slot][count - 1])
                ready.put((slot, count, size))
        except BaseException as exc:
            errors.append(exc)
            ready.put(None)

    producer_thread = threading.Thread(target=producer, daemon=True)
    producer_thread.start()
    transfer = torch.cuda.Stream(device=device)
    offset = torch.tensor(vae.config.latents_mean, device=device, dtype=vae.dtype).view(
        1, -1, 1, 1, 1
    )
    scale = torch.tensor(vae.config.latents_std, device=device, dtype=vae.dtype).view(
        1, -1, 1, 1, 1
    )
    latent = None
    total = cursor = 0
    vae.clear_cache()
    try:
        while True:
            item = ready.get()
            if item is None:
                break
            slot, count, size = item
            with torch.cuda.stream(transfer):
                gpu[slot][:size].copy_(host[slot][:size], non_blocking=True)
            torch.cuda.current_stream(device).wait_stream(transfer)
            pixels = gpu[slot][:size].to(vae.dtype).permute(3, 0, 1, 2)[None].div_(127.5).sub_(1)
            pixels = (
                patchify(pixels, patch_size=vae.config.patch_size)
                if vae.config.patch_size
                else pixels
            )
            vae._enc_conv_idx = [0]
            with torch.profiler.record_function("vae_chunk"):
                if hasattr(vae, "_chunk_runner"):
                    z = vae._chunk_runner(pixels, offset, scale)
                else:
                    encoded = vae.encoder(
                        pixels, feat_cache=vae._enc_feat_map, feat_idx=vae._enc_conv_idx
                    )
                    mean = vae.quant_conv(encoded).chunk(2, 1)[0]
                    z = ((mean - offset) / scale)[0]
            valid_knots = 1 if total == 0 else (count + 3) // 4
            z = z[:, :valid_knots]
            if latent is None:
                latent = torch.empty(
                    (z.shape[0], 1 + (capacity + 2) // 4, *z.shape[-2:]),
                    device=device,
                    dtype=z.dtype,
                )
            if cursor + z.shape[1] > latent.shape[1]:
                raise ValueError("Latent capacity exceeded")
            latent[:, cursor : cursor + z.shape[1]].copy_(z)
            cursor += z.shape[1]
            total += count
            events[slot].record()
            occupied[slot] = True
            free.put(slot)
            if progress and (total < 10 or total % 200 == 1):
                progress("vae", total)
        if errors:
            raise errors[0]
    finally:
        stop.set()
        producer_thread.join(timeout=10)
        if producer_thread.is_alive():
            raise RuntimeError("Video producer failed to stop")
        if hasattr(iterator, "close"):
            iterator.close()
        vae.clear_cache()
    assert cursor == 1 + (total + 2) // 4
    return (latent[:, :cursor], total)


def batch_for(
    first,
    count,
    latent,
    device,
    height=480,
    width=480,
    input_buffers=None,
    *,
    geometry_intrinsics=None,
):
    selection, offset = latent_window(first, count, latent.shape[1])
    window = latent[:, selection]
    if input_buffers is not None:
        shape = tuple(window.shape)
        if shape not in input_buffers:
            input_buffers[shape] = torch.empty(shape, device=device, dtype=window.dtype)
        input_buffers[shape].copy_(window, non_blocking=True)
        window = input_buffers[shape]
    else:
        window = window.to(device).contiguous()
    batch = SimpleNamespace(
        latent_mean=window[None],
        rgb_frames=count,
        temporal_offset=torch.tensor([offset], device=device),
        image_size=torch.tensor([[height, width]], device=device),
    )
    if geometry_intrinsics is not None:
        batch.geometry_intrinsics = torch.as_tensor(
            geometry_intrinsics, device=device, dtype=torch.float32
        ).reshape(1, 3, 3)
    return batch


def combine_window_batches(batches):
    batch = SimpleNamespace(
        rgb_frames=batches[0].rgb_frames,
        **{
            name: torch.cat([getattr(batch, name) for batch in batches], 0)
            for name in ("latent_mean", "temporal_offset", "image_size")
        },
    )
    cameras = [getattr(item, "geometry_intrinsics", None) for item in batches]
    if any(k is not None for k in cameras):
        if any(k is None for k in cameras):
            raise ValueError("Window batches cannot mix predicted and calibrated cameras")
        batch.geometry_intrinsics = torch.cat(cameras, 0)
    return batch


def prediction_item(prediction, index):

    def select(value):
        if value is None:
            return None
        if isinstance(value, dict):
            return {name: select(tensor) for name, tensor in value.items()}
        return value[index : index + 1]

    return type(prediction)(
        **{field.name: select(getattr(prediction, field.name)) for field in fields(prediction)}
    )


def window_groups(windows, latent_frames, batch_size):
    if batch_size < 1:
        raise ValueError("window_batch_size must be positive")
    pending, key = ([], None)

    def capacity(group):
        return len(group)

    for window in windows:
        selection, offset = latent_window(*window, latent_frames)
        current = (selection.stop - selection.start, offset, window[1])
        if pending and (current != key or len(pending) == batch_size):
            yield (pending, capacity(pending))
            pending = []
        pending.append(window)
        key = current
    if pending:
        yield (pending, capacity(pending))


def iter_window_predictions(
    net, windows, latent, device, height, width, batch_size, *, geometry_intrinsics=None
):
    buffers = {}
    if geometry_intrinsics is not None:
        geometry_intrinsics = torch.as_tensor(
            geometry_intrinsics, device=device, dtype=torch.float32
        )
    for group, size in window_groups(windows, latent.shape[1], batch_size):
        if size == 1:
            batch = batch_for(
                *group[0],
                latent,
                device,
                height,
                width,
                buffers,
                geometry_intrinsics=geometry_intrinsics,
            )
        else:
            batches = [
                batch_for(
                    *window,
                    latent,
                    device,
                    height,
                    width,
                    geometry_intrinsics=geometry_intrinsics,
                )
                for window in group
            ]
            batch = combine_window_batches(batches)
        with (
            torch.profiler.record_function("backbone_readout_pnp_mano"),
            torch.autocast(device.type, dtype=torch.bfloat16),
        ):
            prediction = net(batch)
        for index, window in enumerate(group):
            yield (window, prediction if size == 1 else prediction_item(prediction, index))


@torch.inference_mode()
def infer_video(
    net,
    vae,
    source,
    device,
    *,
    max_frames=0,
    expected_frames=85,
    progress=None,
    width=480,
    height=480,
    window_batch_size=1,
    resize_mode="center_crop",
    long_edge=832,
    direct_pinned=False,
    return_latent=True,
    geometry_intrinsics=None,
):
    """Infer with optional fixed pixel K in the processed canvas coordinate system."""
    geometry_intrinsics = validate_intrinsics(geometry_intrinsics)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    metadata = {}
    plan = resize_plan(
        probe_video(source)["source_size"],
        resize_mode,
        width=width,
        height=height,
        long_edge=long_edge,
    )
    width, height = plan["size"]
    if direct_pinned:
        from .pinned_video import PinnedFrames, encode_pinned

        frames = PinnedFrames(source, plan, metadata, max_frames)
        latent, total = encode_pinned(
            vae, frames, device, max_frames or expected_frames, progress=progress
        )
    else:
        latent, total = encode_double_buffer(
            vae,
            ffmpeg_frames(source, plan, metadata, max_frames),
            device,
            max_frames or expected_frames,
            progress=progress,
        )
    torch.cuda.synchronize(device)
    encoded = time.perf_counter()
    windows = sequence_windows(total, 81, 40)
    means = torch.stack((net.mano.left.pose_mean, net.mano.right.pose_mean))
    fusion = SequenceFusion(total, means, storage_device=device)
    ksum = torch.zeros((total, 3, 3), device=device, dtype=torch.float64)
    kcount = torch.zeros(total, device=device, dtype=torch.float64)
    heatmaps, knots = ([], [])
    predictions = iter_window_predictions(
        net,
        windows,
        latent,
        device,
        height,
        width,
        window_batch_size,
        geometry_intrinsics=geometry_intrinsics,
    )
    for ordinal, ((first, count), prediction) in enumerate(predictions):
        fusion.add(first, prediction)
        camera = torch.eye(3, device=device, dtype=torch.float64)
        camera[0, 0], camera[1, 1] = prediction.camera_fit["focal"][0].double() * torch.tensor(
            [width, height], device=device
        )
        camera[:2, 2] = prediction.camera_fit["center"][0].double() * torch.tensor(
            [width, height], device=device
        )
        weight = torch.minimum(
            torch.arange(count, device=device) + 1, count - torch.arange(count, device=device)
        )
        ksum[first : first + count] += camera * weight[:, None, None]
        kcount[first : first + count] += weight
        heatmaps.append(prediction.joint_heatmaps[0].detach())
        knots.append(prediction.heatmap_frame_indices[0].detach())
        if progress and (ordinal % 10 == 0 or ordinal == len(windows) - 1):
            progress("window", ordinal + 1)
    torch.cuda.synchronize(device)
    decoded = time.perf_counter()
    if progress:
        progress("fusion", 0)
    prediction = fusion.finish(net.mano, device, chunk_size=512)
    torch.cuda.synchronize(device)
    fused = time.perf_counter()
    if progress:
        progress("transfer", 0)
    result = {name: getattr(prediction, name)[0].detach().float().cpu().numpy() for name in FIELDS}
    result.update(
        frame_indices=np.arange(total),
        timestamps=np.array(metadata["timestamps"]),
        predicted_intrinsics=(ksum / kcount[:, None, None]).float().cpu().numpy(),
        window_starts=np.array([start for start, count in windows]),
        window_counts=np.array([count for start, count in windows]),
    )
    result["overlay_intrinsics"] = result["predicted_intrinsics"].copy()
    if geometry_intrinsics is not None:
        result["predicted_intrinsics"] = np.broadcast_to(
            geometry_intrinsics.astype(np.float32), (total, 3, 3)
        ).copy()
        result["overlay_intrinsics"] = result["predicted_intrinsics"].copy()
    result["predicted_camera_valid"] = np.ones(total, dtype=bool)
    result["overlay_camera_valid"] = np.ones(total, dtype=bool)
    maximum = max((value.shape[0] for value in heatmaps))
    result["joint_heatmaps"] = (
        torch.stack(
            [
                torch.cat((value, value.new_zeros(maximum - value.shape[0], *value.shape[1:])))
                for value in heatmaps
            ]
        )
        .float()
        .cpu()
        .numpy()
    )
    result["heatmap_frame_indices"] = (
        torch.stack(
            [torch.cat((value, value.new_full((maximum - value.shape[0],), -1))) for value in knots]
        )
        .cpu()
        .numpy()
    )
    if return_latent:
        result["latent_mean"] = latent.float().cpu().numpy()
    if getattr(net, "_deferred_camera_checks", None):
        if not bool(torch.cat(net._deferred_camera_checks).all()):
            raise ValueError("Nonfinite camera parameters")
        net._deferred_camera_checks = []
    torch.cuda.synchronize(device)
    finished = time.perf_counter()
    for name, value in result.items():
        if not np.isfinite(value).all():
            raise ValueError(f"Nonfinite output: {name}")
    metadata.update(
        frames=total,
        size=[width, height],
        windows=windows,
        window=81,
        stride=40,
        window_batch_size=window_batch_size,
        vae_chunk_frames=getattr(vae, "_acceleration_chunk_frames", 4),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        timings=dict(
            vae_decode_preprocess_seconds=encoded - start,
            windows_seconds=decoded - encoded,
            fusion_mano_seconds=fused - decoded,
            final_cpu_transfer_seconds=finished - fused,
            inference_seconds=finished - start,
            fps=total / (finished - start),
        ),
    )
    metadata.update(
        geometry_camera_source="calibration" if geometry_intrinsics is not None else "predicted",
        geometry_intrinsics_canvas=(
            geometry_intrinsics.tolist() if geometry_intrinsics is not None else None
        ),
        decoder_and_pnp_use_calibration=geometry_intrinsics is not None,
    )
    return (result, metadata)
