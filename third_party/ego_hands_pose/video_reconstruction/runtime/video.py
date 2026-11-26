from fractions import Fraction

import av
import numpy as np


class VideoWriter:
    def __init__(self, path, width, height, fps, *, codec="libx264", faststart=False):
        if codec not in {"libx264", "h264_nvenc", "auto"}:
            raise ValueError(f"Unknown overlay encoder: {codec}")
        self.container = None
        self.fallback_reason = None
        candidates = ("h264_nvenc", "libx264") if codec == "auto" else (codec,)
        for candidate in candidates:
            try:
                self.container = av.open(
                    str(path), "w", options={"movflags": "+faststart"} if faststart else {}
                )
                self.stream = self.container.add_stream(
                    candidate, rate=Fraction(str(fps)).limit_denominator(100000)
                )
                self.stream.width, self.stream.height = (width, height)
                self.stream.pix_fmt = "yuv420p"
                self.stream.options = (
                    {
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
                    if candidate == "h264_nvenc"
                    else {"crf": "18", "preset": "fast"}
                )
                self.stream.time_base = Fraction(1, 90000)
                self.stream.codec_context.time_base = Fraction(1, 90000)
                self.options = dict(self.stream.options)
                self.stream.codec_context.open()
                self.codec = candidate
                break
            except (av.error.FFmpegError, ValueError) as error:
                if self.container is not None:
                    self.container.close()
                    self.container = None
                if candidate == candidates[-1]:
                    raise
                self.fallback_reason = f"{type(error).__name__}: {error}"
        self.last_pts = -1

    def write(self, rgb, timestamp):
        frame = av.VideoFrame.from_ndarray(np.asarray(rgb), format="rgb24")
        pts = max(self.last_pts + 1, round(timestamp * 90000))
        frame.pts, frame.time_base = (pts, Fraction(1, 90000))
        self.last_pts = pts
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self):
        if self.container is not None:
            try:
                for packet in self.stream.encode():
                    self.container.mux(packet)
            finally:
                self.container.close()
                self.container = None
