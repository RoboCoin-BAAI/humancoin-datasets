from fractions import Fraction
from functools import lru_cache
from itertools import chain, islice
from time import perf_counter

import av
import numpy as np
from PIL import Image, ImageDraw


class GPUOverlayUnavailable(RuntimeError):
    pass


class GPUOverlayEncodingError(RuntimeError):
    pass


@lru_cache(maxsize=2)
def _cuda_frame_context(device):
    return av.video.frame.CudaContext(device_id=device, primary_ctx=False)


def clear_cuda_context_cache():
    _cuda_frame_context.cache_clear()


def open_decoder(path, device, size):
    from av.codec.hwaccel import HWAccel

    fallback = None
    container = None
    try:
        container = av.open(
            str(path),
            hwaccel=HWAccel(
                "cuda",
                device=str(device),
                is_hw_owned=True,
                allow_software_fallback=False,
                options={"primary_ctx": "0", "current_ctx": "1"},
            ),
        )
        frames = container.decode(video=0)
        first = next(frames)
        codec = container.streams.video[0].codec_context
        if not codec.is_hwaccel or first.format.name != "cuda":
            decoder = "cpu"
            fallback = f"NVDEC probe selected software decoder for {codec.name}/{first.format.name}"
        elif (
            first.sw_format.name != "nv12"
            or first.colorspace not in (1, 2, 5, 6)
            or first.color_range not in (0, 1, 2)
        ):
            raise ValueError(
                f"Unsupported zero-copy surface/color: {first.format.name}, {first.sw_format}, matrix={first.colorspace}, range={first.color_range}"
            )
        else:
            decoder = "nvdec"
    except (av.error.FFmpegError, ValueError, RuntimeError, StopIteration) as error:
        if container is not None:
            container.close()
        fallback = f"{type(error).__name__}: {error}"
        container = av.open(str(path))
        frames = container.decode(video=0)
        try:
            first = next(frames)
        except StopIteration:
            container.close()
            raise ValueError("Empty overlay background video") from None
        decoder = "cpu"
    if [first.width, first.height] != list(size):
        container.close()
        raise ValueError("Overlay background dimensions do not match requested canvas")
    return (container, chain((first,), frames), first, decoder, fallback)


def _planes(frame, *, software=False):
    import torch

    if not software:
        return tuple((torch.utils.dlpack.from_dlpack(p.__dlpack__()) for p in frame.planes))
    if frame.format.name != "nv12":
        frame = frame.reformat(format="nv12")
    y, uv = frame.planes
    yp = np.frombuffer(y, np.uint8).reshape(y.height, y.line_size)[:, : frame.width].copy()
    up = np.frombuffer(uv, np.uint8).reshape(uv.height, uv.line_size)[:, : frame.width].copy()
    return (
        torch.from_numpy(yp),
        torch.from_numpy(up.reshape(frame.height // 2, frame.width // 2, 2)),
    )


class GPUWriter:
    def __init__(self, path, width, height, fps, device, matrix, color_range):
        from av.codec.hwaccel import HWAccel

        self.container = None
        try:
            self.container = av.open(
                str(path),
                "w",
                options={
                    "movflags": "+faststart+negative_cts_offsets",
                    "movie_timescale": "90000",
                    "avoid_negative_ts": "disabled",
                },
            )
            self.stream = self.container.add_stream(
                "h264_nvenc",
                rate=Fraction(str(fps)).limit_denominator(100000),
                hwaccel=HWAccel(
                    "cuda",
                    device=str(device),
                    is_hw_owned=True,
                    options={"primary_ctx": "0", "current_ctx": "1"},
                ),
            )
            self.stream.width, self.stream.height = (width, height)
            self.stream.pix_fmt = "nv12"
            self.options = {
                "preset": "p4",
                "tune": "hq",
                "rc": "vbr",
                "cq": "18",
                "b": "0",
                "spatial-aq": "1",
                "temporal-aq": "1",
                "rc-lookahead": "20",
                "bf": "3",
            }
            self.stream.options = self.options
            self.stream.time_base = self.stream.codec_context.time_base = Fraction(1, 90000)
            self.stream.codec_context.colorspace = matrix
            self.stream.codec_context.color_range = color_range
            self.stream.codec_context.open()
        except Exception as error:
            if self.container is not None:
                self.container.close()
            raise GPUOverlayUnavailable(f"Zero-copy NVENC unavailable: {error}") from error
        self.width, self.height, self.device = (width, height, device)
        self.matrix, self.color_range = (matrix, color_range)
        self.last_pts = -1
        self.closed = False
        self.failed = False
        self.submitted = 0
        self.cuda_context = _cuda_frame_context(device)

    def _encode(self, frame=None):
        try:
            return self.stream.encode(frame)
        except av.error.FFmpegError as error:
            self.failed = True
            stage = f"frame {self.submitted}" if frame is not None else "flush"
            detail = getattr(error, "log", None)
            raise GPUOverlayEncodingError(
                f"NVENC CUDA input failed at {stage}: {error}; FFmpeg: {detail}"
            ) from error

    def write(self, y, uv, timestamp):
        if (
            y.stride(0) != self.width
            or uv.stride(0) != self.width
            or uv.data_ptr() != y.data_ptr() + self.height * self.width
        ):
            raise ValueError("NVENC requires contiguous pitched Y/UV planes in one NV12 allocation")
        frame = av.VideoFrame.from_dlpack(
            (y, uv),
            format="nv12",
            width=self.width,
            height=self.height,
            device_id=self.device,
            stream=None,
            primary_ctx=False,
            cuda_context=self.cuda_context,
        )
        pts = max(self.last_pts + 1, round(timestamp * 90000))
        self.last_pts = pts
        frame.pts, frame.time_base = (pts, Fraction(1, 90000))
        frame.colorspace, frame.color_range = (self.matrix, self.color_range)
        for packet in self._encode(frame):
            self.container.mux(packet)
        self.submitted += 1
        return frame

    def close(self, *, abort=False):
        if not self.closed:
            self.closed = True
            try:
                if not (abort or self.failed):
                    for packet in self._encode():
                        self.container.mux(packet)
            except BaseException as error:
                try:
                    self.container.close()
                except BaseException as cleanup:
                    error.add_note(f"NVENC container cleanup also failed: {cleanup}")
                raise
            else:
                self.container.close()


def _geometry(result, meta, first, count, faces, threshold, mode, source_canvas):
    from .overlay import canvas_to_source, project

    stop = first + count
    active = result["existence_prob"][first:stop] >= threshold
    if mode != "anchors":
        active &= result["overlay_camera_valid"][first:stop, None]
    joints = []
    vertices = []
    valids = []
    triangle_valids = []
    for i in range(first, stop):
        for hand in (0, 1):
            if mode == "anchors":
                uv = result["anchors_2d"][i, hand] * meta["size"]
                valid = np.isfinite(uv).all(-1)
            else:
                uv, valid = project(result["joints_3d"][i, hand], result["overlay_intrinsics"][i])
            if source_canvas:
                uv = canvas_to_source(uv, meta["crop_affine"])
            valid &= np.isfinite(uv).all(-1) & (np.abs(uv).max(-1) < 10000)
            valid &= active[i - first, hand]
            joints.append(uv)
            valids.append(valid)
            if mode == "mesh":
                v, good = project(result["vertices"][i, hand], result["overlay_intrinsics"][i])
                if source_canvas:
                    v = canvas_to_source(v, meta["crop_affine"])
                triangles = np.asarray(faces[hand], dtype=np.int64)
                poly = v[triangles]
                tv = good[triangles].all(-1) & np.isfinite(poly).all((-1, -2))
                tv &= (np.abs(poly).max((-1, -2)) <= 10000) & active[i - first, hand]
                depths = result["vertices"][i, hand, triangles, 2].mean(-1)
                order = np.argsort(depths)[::-1]
                rank = np.empty(len(order))
                rank[order] = np.arange(len(order))
                z = 0.9 - 1.8 * rank / max(len(order) - 1, 1)
                vertices.append((poly, z))
                triangle_valids.append(tv)
    return (joints, valids, vertices, triangle_valids)


def _layers(
    result,
    meta,
    first,
    count,
    faces,
    threshold,
    mode,
    width,
    height,
    source_canvas,
    ctx,
    device,
    font,
    label_bases,
):
    import torch

    from .overlay import EDGES
    from .overlay_indicators import COLORS, bar_panels, concentration_colors, indicator_label
    from .overlay_raster import _paint_labels, _paint_skeleton, label_layout

    geometry = _geometry(result, meta, first, count, faces, threshold, mode, source_canvas)
    joints, valids, vertices, triangle_valids = geometry
    points = [uv[valid] for uv, valid in zip(joints, valids) if valid.any()]
    points += [
        poly[good].reshape(-1, 2)
        for (poly, _), good in zip(vertices, triangle_valids)
        if good.any()
    ]
    if points:
        allpoints = np.concatenate(points)
        margin = (
            max(4, round(3 * min(width, height) / 480) + 1)
            if "overlay_joint_concentration" in result
            else 4
        )
        left, top = np.floor(allpoints.min(0)).astype(int) - margin
        right, bottom = np.ceil(allpoints.max(0)).astype(int) + margin + 1
        left, top = np.clip([left, top], 0, [width - 1, height - 1])
        right, bottom = np.clip([right, bottom], [left + 1, top + 1], [width, height])
    else:
        left = top = 0
        right = bottom = 1
    rw = int((right - left + 7) // 8 * 8)
    rh = int((bottom - top + 7) // 8 * 8)
    box = (int(left), int(top), rw, rh)
    masks = []
    for uv, valid in zip(joints, valids):
        mask = Image.new("L", (width, height))
        _paint_skeleton(
            ImageDraw.Draw(mask),
            uv,
            valid,
            EDGES,
            1,
            width,
            height,
            list(range(2, 23)) if "overlay_joint_concentration" in result else None,
        )
        masks.append(np.asarray(mask.crop((left, top, left + rw, top + rh)), dtype=np.uint8))
    skeleton = torch.as_tensor(np.stack(masks), device=device)
    palette = np.tile(np.asarray(COLORS, np.uint8)[None, :, None, :], (count, 1, 23, 1))
    if "overlay_joint_concentration" in result:
        palette[:, :, 2:] = concentration_colors(
            result["overlay_joint_concentration"][first : first + count]
        )
    palette = torch.as_tensor(palette, device=device)
    rast = altitudes = None
    if mode == "mesh":
        import nvdiffrast.torch as dr

        if len(faces[0]) != len(faces[1]):
            raise ValueError("Batched overlay requires equal left/right topology sizes")
        nfaces = len(faces[0])
        positions = []
        heights = []
        for (poly, z), good in zip(vertices, triangle_valids):
            pos = np.empty((nfaces, 3, 4), np.float32)
            pos[..., :2] = 2 * (poly - [left, top] + 0.5) / [rw, rh] - 1
            pos[..., 2] = z[:, None]
            pos[..., 3] = 1
            pos[~good] = [0, 0, 0, 1]
            positions.append(pos.reshape(-1, 4))
            edges = poly[:, [1, 2, 0]] - poly[:, [2, 0, 1]]
            area = np.abs(
                (poly[:, 1, 0] - poly[:, 0, 0]) * (poly[:, 2, 1] - poly[:, 0, 1])
                - (poly[:, 1, 1] - poly[:, 0, 1]) * (poly[:, 2, 0] - poly[:, 0, 0])
            )
            heights.append(
                np.nan_to_num(area[:, None] / np.maximum(np.linalg.norm(edges, axis=-1), 1e-08))
            )
        pos = torch.as_tensor(np.stack(positions), device=device)
        tri = torch.arange(nfaces * 3, dtype=torch.int32, device=device).reshape(-1, 3)
        altitudes = torch.as_tensor(np.stack(heights).astype(np.float32), device=device)
        rast, _ = dr.rasterize(ctx, pos, tri, resolution=[rh, rw], grad_db=False)
    labels = []
    prefix = f"Reconstruction | {mode} | L green / R pink | Frame "
    size_key = ("canvas_size", mode)
    if size_key not in label_bases:
        digit = max("0123456789", key=font.getlength)
        ids = result["frame_indices"]
        id_length = max(len(str(int(ids[0]))), len(str(int(ids[-1]))))
        times = meta["timestamps"]
        time_length = max(len(f"{times[0]:.3f}"), len(f"{times[-1]:.3f}"))
        template = prefix + digit * id_length + " | " + digit * time_length + "s"
        camera = (
            "camera: supplied"
            if meta["overlay_camera"] == "supplied"
            else "camera: predicted pinhole"
            if meta["overlay_camera"] == "predicted_parametric_pinhole"
            else "camera: estimated (approximate)"
        )
        hint = indicator_label(result)
        options = (
            [hint]
            if mode == "anchors"
            else [camera + " | " + hint, camera + " / invalid; skipped | " + hint]
        )
        layouts = [label_layout(template, text, font, width, height) for text in options]
        label_bases[size_key] = (
            width
            if any((l["title_lines"] > 1 for l in layouts))
            else max((l["width"] for l in layouts)),
            max((l["height"] for l in layouts)),
        )
    tw, th = label_bases[size_key]
    for i in range(first, first + count):
        label = None
        if mode != "anchors":
            label = (
                "camera: supplied"
                if meta["overlay_camera"] == "supplied"
                else "camera: predicted pinhole"
                if meta["overlay_camera"] == "predicted_parametric_pinhole"
                else "camera: estimated (approximate)"
            )
            if not result["overlay_camera_valid"][i]:
                label += " / invalid; skipped"
        hint = indicator_label(result)
        label = hint if label is None else label + " | " + hint
        tail = f"{int(result['frame_indices'][i])} | {meta['timestamps'][i]:.3f}s"
        layout = label_layout(prefix + tail, label, font, width, height)
        key = (mode, label, tw, th)
        if layout["title_lines"] == 1:
            if key not in label_bases:
                base = Image.new("RGBA", (tw, th))
                ImageDraw.Draw(base).rectangle(
                    (0, 0, tw - 1, layout["height"] - 1), fill=(12, 15, 20)
                )
                _paint_labels(ImageDraw.Draw(base), prefix, label, font, width, height)
                label_bases[key] = (base, layout["padding"] + font.getlength(prefix))
            base, tail_x = label_bases[key]
            img = base.copy()
            if layout["width"] < tw:
                ImageDraw.Draw(img).rectangle(
                    (layout["width"], 0, tw - 1, th - 1), fill=(0, 0, 0, 0)
                )
            ImageDraw.Draw(img).text((tail_x, layout["padding"]), tail, fill="white", font=font)
        else:
            img = Image.new("RGBA", (tw, th))
            _paint_labels(ImageDraw.Draw(img), prefix + tail, label, font, width, height)
        labels.append(np.asarray(img))
    text = torch.as_tensor(np.stack(labels), device=device)
    panels = [
        bar_panels(result["existence_prob"][i], width, height, font)
        for i in range(first, first + count)
    ]
    bars = torch.as_tensor(np.stack([item[0] for item in panels]), device=device)
    return (rast, altitudes, skeleton, text, box, palette, bars, panels[0][1])


def render_cuda(
    input_video,
    output_video,
    result,
    metadata,
    faces,
    threshold,
    mode,
    progress=None,
    *,
    batch_size=2,
):
    try:
        import torch

        from .cuda_video_memory import nvenc_buffer
        from .gpu_overlay_kernels import composite_nv12
    except ImportError as error:
        raise GPUOverlayUnavailable(f"CUDA overlay dependency unavailable: {error}") from error
    started = perf_counter()
    from .overlay_indicators import prepare_indicators

    result = prepare_indicators(result, metadata)
    if not torch.cuda.is_available():
        raise GPUOverlayUnavailable("CUDA is unavailable")
    if not hasattr(av.VideoFrame, "from_dlpack"):
        raise GPUOverlayUnavailable("PyAV with CUDA DLPack support is required")
    width, height = metadata["source_size"]
    if width % 2 or height % 2 or mode not in ("anchors", "joints", "mesh"):
        raise GPUOverlayUnavailable(
            "CUDA NV12 overlay requires even dimensions and a supported mode"
        )
    if not isinstance(batch_size, int) or not 1 <= batch_size <= 16:
        raise ValueError("GPU overlay batch size must be between 1 and 16")
    device = torch.device(metadata.get("device", "cuda:0"))
    if device.type != "cuda":
        raise GPUOverlayUnavailable("Prediction device is not CUDA")
    device_id = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.set_device(device_id)
    free, _ = torch.cuda.mem_get_info(device_id)
    affordable = int(max(0, free - 2 * 1024**3) // max(width * height * 80, 1))
    if affordable < 1:
        raise GPUOverlayUnavailable("Insufficient free GPU memory for bounded overlay workspace")
    batch_size = min(batch_size, affordable)
    ctx = None
    if mode == "mesh":
        try:
            import nvdiffrast.torch as dr

            ctx = dr.RasterizeCudaContext(device=device)
        except Exception as error:
            raise GPUOverlayUnavailable(f"nvdiffrast unavailable: {error}") from error
    container, frames, first_frame, decoder, reason = open_decoder(
        input_video, device_id, (width, height)
    )
    matrix = first_frame.colorspace if first_frame.colorspace in (1, 5, 6) else 6
    color_range = first_frame.color_range if first_frame.color_range in (1, 2) else 1
    if decoder == "cpu" and first_frame.colorspace not in (1, 2, 5, 6):
        container.close()
        raise GPUOverlayUnavailable(
            "Unsupported source color matrix; retain reference RGB conversion"
        )
    source_matrix, source_range = (first_frame.colorspace, first_frame.color_range)
    del first_frame
    try:
        writer = GPUWriter(
            output_video, width, height, metadata["fps"], device_id, matrix, color_range
        )
    except BaseException:
        container.close()
        raise
    from .overlay_raster import label_font

    font = label_font(width, height)
    label_bases = {}
    count = len(result["frame_indices"])
    stages = dict(
        decode_seconds=0.0,
        geometry_raster_composite_seconds=0.0,
        encode_seconds=0.0,
        capture_seconds=0.0,
    )
    rendered = 0
    last_progress = 0
    try:
        with torch.cuda.device(device_id), torch.inference_mode():
            while rendered < count:
                start = perf_counter()
                batch = list(islice(frames, min(batch_size, count - rendered)))
                if not batch:
                    raise ValueError("Overlay frame count mismatch")
                b = len(batch)
                nv12 = nvenc_buffer(b, height, width, device)
                y = nv12[:, :height]
                uv = nv12[:, height:].reshape(b, height // 2, width // 2, 2)
                for i, frame in enumerate(batch):
                    if (frame.width, frame.height) != (width, height):
                        raise ValueError("Overlay background dimensions changed")
                    if frame.colorspace != source_matrix or frame.color_range != source_range:
                        raise ValueError("Overlay background color contract changed")
                    yp, up = _planes(frame, software=decoder == "cpu")
                    y[i].copy_(yp)
                    uv[i].copy_(up)
                stages["decode_seconds"] += perf_counter() - start
                start = perf_counter()
                layers = _layers(
                    result,
                    metadata,
                    rendered,
                    b,
                    faces,
                    threshold,
                    mode,
                    width,
                    height,
                    True,
                    ctx,
                    device,
                    font,
                    label_bases,
                )
                composite_nv12(y, uv, *layers, matrix=matrix, full=color_range == 2)
                done = torch.cuda.Event()
                done.record()
                done.synchronize()
                del batch, frame, yp, up
                stages["geometry_raster_composite_seconds"] += perf_counter() - start
                for i in range(b):
                    index = rendered + i
                    start = perf_counter()
                    writer.write(y[i], uv[i], metadata["timestamps"][index])
                    stages["encode_seconds"] += perf_counter() - start
                rendered += b
                if progress and (
                    rendered == b or rendered == count or rendered - last_progress >= 30
                ):
                    progress(rendered)
                    last_progress = rendered
        start = perf_counter()
        writer.close()
        stages["encode_seconds"] += perf_counter() - start
    except BaseException as error:
        try:
            writer.close(abort=True)
        except BaseException as cleanup:
            error.add_note(f"NVENC abort also failed: {cleanup}")
        try:
            container.close()
        except BaseException as cleanup:
            error.add_note(f"Overlay decoder cleanup also failed: {cleanup}")
        raise
    else:
        container.close()
    metadata["overlay_render"] = dict(
        backend="cuda_nv12",
        mesh_rasterizer="nvdiffrast" if mode == "mesh" else None,
        decoder=decoder,
        decoder_fallback_reason=reason,
        encoder="h264_nvenc",
        encoder_options=writer.options,
        encoder_fallback_reason=None,
        compression="NVENC CQ18",
        pixel_format="nv12",
        frames=rendered,
        seconds=perf_counter() - started,
        batch_size=batch_size,
        stage_seconds_overlap=False,
        color_matrix=matrix,
        color_range=color_range,
        mesh_depth="original_mean_triangle_depth_order",
        visual_changes=["GPU triangle coverage/edge pixels", "YUV alpha/chroma rounding"],
        nvenc_buffer_allocator="cuMemAlloc",
        full_frame_cpu_transfer=False,
        prediction_modified=False,
        **stages,
    )
    return rendered
