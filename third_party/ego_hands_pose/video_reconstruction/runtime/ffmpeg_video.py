import json
import math
import queue
import re
import subprocess
import threading
from collections import deque
from fractions import Fraction

import numpy as np


def count_video_frames(path):
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-threads",
        "2",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode or result.stderr.strip():
        raise RuntimeError("Video frame counting failed: " + result.stderr.strip())
    streams = json.loads(result.stdout).get("streams", [])
    try:
        count = int(streams[0]["nb_read_frames"])
    except (IndexError, KeyError, TypeError, ValueError):
        raise ValueError("No decodable video frames found while counting") from None
    if count <= 0:
        raise ValueError("No decodable video frames found while counting")
    return count


def probe_video(path):
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_streams",
        "-of",
        "json",
        str(path),
    ]
    info = json.loads(subprocess.check_output(command, text=True))["streams"]
    if not info:
        raise ValueError("No video stream")
    stream = info[0]
    if any((int(d.get("rotation", 0)) % 360 for d in stream.get("side_data_list", []))):
        raise ValueError("Rotated video requires an explicit source-coordinate convention")
    if stream.get("sample_aspect_ratio", "1:1") not in ("1:1", "0:1", "N/A"):
        raise ValueError("Non-square source pixels require an explicit camera transform")
    rate = stream.get("avg_frame_rate", "0/0")
    if rate in ("0/0", "0/1"):
        rate = stream.get("r_frame_rate", "30/1")
    return dict(source_size=[stream["width"], stream["height"]], fps=float(Fraction(rate)))


def resize_plan(source_size, mode="long_edge_pad", *, long_edge=832, width=672, height=480):
    sw, sh = map(int, source_size)
    if min(sw, sh) <= 0:
        raise ValueError("Invalid video size")
    left = top = pad_x = pad_y = 0
    if mode == "long_edge_pad":
        if long_edge < 64 or long_edge % 32:
            raise ValueError("Long edge must be a multiple of 32")
        scale = long_edge / max(sw, sh)
        rw, rh = (max(1, round(sw * scale)), max(1, round(sh * scale)))
        width, height = (math.ceil(rw / 32) * 32, math.ceil(rh / 32) * 32)
        pad_x, pad_y = ((width - rw) // 2, (height - rh) // 2)
        cw, ch = (sw, sh)
        filters = [
            f"scale={rw}:{rh}:flags=bilinear",
            "format=rgb24",
            f"pad={width}:{height}:{pad_x}:{pad_y}:color=black",
        ]
    elif mode == "center_crop":
        if width % 32 or height % 32 or min(width, height) < 64:
            raise ValueError("Canvas must be multiples of 32")
        ch, cw = (
            (sh, round(sh * width / height))
            if sw / sh > width / height
            else (round(sw * height / width), sw)
        )
        left, top = ((sw - cw) // 2, (sh - ch) // 2)
        rw, rh = (width, height)
        filters = [
            "format=rgb24",
            f"crop={cw}:{ch}:{left}:{top}:exact=1",
            f"scale={rw}:{rh}:flags=bilinear",
        ]
    else:
        raise ValueError(f"Unknown resize mode: {mode}")
    sx, sy = (rw / cw, rh / ch)
    tx, ty = (pad_x - sx * left + (sx - 1) / 2, pad_y - sy * top + (sy - 1) / 2)
    matrix = np.array([[sx, 0, tx], [0, sy, ty], [0, 0, 1.0]])
    return dict(
        size=[width, height],
        source_size=[sw, sh],
        resize_mode=mode,
        resized_content_size=[rw, rh],
        padding=[pad_x, pad_y, width - rw - pad_x, height - rh - pad_y],
        content_rect=[pad_x, pad_y, pad_x + rw, pad_y + rh],
        source_crop=[left, top, cw, ch],
        source_to_canvas=matrix.tolist(),
        canvas_to_source_matrix=np.linalg.inv(matrix).tolist(),
        crop_affine=[-tx / sx, -ty / sy, sx, sy],
        coordinate_convention="pixel_centers_half_pixel_resize",
        ffmpeg_filter=",".join(filters + ["setsar=1"]),
        expected_latent_grid=[height // 16, width // 16],
    )


def ffmpeg_frames(path, plan, metadata, max_frames=0, *, probe=None, buffer_provider=None):
    probe = probe_video(path) if probe is None else probe
    if probe["source_size"] != plan["source_size"]:
        raise ValueError("Source dimensions changed")
    metadata.update(
        plan,
        fps=probe["fps"],
        timestamps=[],
        truncated=False,
        video_decoder="ffmpeg_rawvideo_pipe",
        intermediate_video=False,
    )
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "info",
        "-noautorotate",
        "-threads",
        "2",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-vf",
        plan["ffmpeg_filter"] + ",showinfo",
        "-fps_mode",
        "passthrough",
    ]
    if max_frames:
        command += ["-frames:v", str(max_frames + 1)]
    command += [
        "-c:v",
        "rawvideo",
        "-threads",
        "1",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    metadata["ffmpeg_command"] = command
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stamps = queue.Queue()
    messages = deque(maxlen=20)

    def read_stderr():
        time_base = None
        try:
            for raw in iter(proc.stderr.readline, b""):
                line = raw.decode("utf-8", errors="replace")
                messages.append(line.strip())
                if "showinfo" not in line:
                    continue
                match = re.search("config in time_base:\\s*(\\d+/\\d+)", line)
                if match:
                    time_base = Fraction(match.group(1))
                match = re.search("\\bn:\\s*(\\d+)\\s+pts:\\s*(-?\\d+)\\s+pts_time:", line)
                if match:
                    if time_base is None:
                        raise ValueError("Missing filter time base")
                    stamps.put((int(match.group(1)), float(int(match.group(2)) * time_base)))
        except Exception as exc:
            stamps.put(exc)
        finally:
            stamps.put(None)

    reader = threading.Thread(target=read_stderr, daemon=True)
    reader.start()
    width, height = plan["size"]
    frame_bytes = width * height * 3
    origin = None
    index = 0
    complete = False
    try:
        while True:
            if buffer_provider is None:
                raw = proc.stdout.read(frame_bytes)
            else:
                raw = memoryview(buffer_provider()).cast("B")
                if len(raw) != frame_bytes or raw.readonly:
                    raise ValueError("FFmpeg target must be a writable RGB frame buffer")
                filled = 0
                while filled < frame_bytes:
                    received = proc.stdout.readinto(raw[filled:])
                    if not received:
                        break
                    filled += received
                raw = raw[:filled]
            if not raw:
                break
            if len(raw) != frame_bytes:
                raise ValueError("Partial FFmpeg RGB frame")
            stamp = stamps.get(timeout=30)
            if isinstance(stamp, Exception):
                raise stamp
            if stamp is None or stamp[0] != index:
                raise ValueError("FFmpeg frame/PTS mismatch")
            if max_frames and index >= max_frames:
                metadata["truncated"] = True
                break
            if origin is None:
                origin = stamp[1]
            timestamp = stamp[1] - origin
            if metadata["timestamps"] and timestamp <= metadata["timestamps"][-1]:
                raise ValueError("Non-increasing decoded timestamps")
            metadata["timestamps"].append(timestamp)
            yield np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
            index += 1
        complete = True
    finally:
        proc.stdout.close()
        if not complete or metadata["truncated"]:
            if proc.poll() is None:
                proc.terminate()
        try:
            returncode = proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise RuntimeError("FFmpeg failed to shut down") from None
        reader.join(timeout=5)
        proc.stderr.close()
    if returncode and (not metadata["truncated"]):
        raise RuntimeError("FFmpeg decoding failed: " + "\n".join(messages))
    if not metadata["timestamps"]:
        raise ValueError("No decodable video frames")
