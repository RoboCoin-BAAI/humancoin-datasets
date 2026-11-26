import queue
import threading

import torch

from .ffmpeg_video import ffmpeg_frames


class PinnedFrames:
    def __init__(self, source, plan, metadata, max_frames=0):
        self.plan = plan
        self.destination = None
        self.iterator = ffmpeg_frames(
            source, plan, metadata, max_frames, buffer_provider=lambda: self.destination
        )

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.iterator)

    def close(self):
        self.iterator.close()


@torch.inference_mode()
def encode_pinned(vae, frames, device, capacity, *, progress=None):
    from diffusers.models.autoencoders.autoencoder_kl_wan import patchify

    chunk_frames = getattr(vae, "_acceleration_chunk_frames", 4)
    if chunk_frames < 4 or chunk_frames % 4:
        raise ValueError("VAE chunk size must be a positive multiple of four")
    iterator = iter(frames)
    w, h = frames.plan["size"]
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
                        frames.destination = host[slot][index].numpy()
                        next(iterator)
                    except StopIteration:
                        break
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
