"""Rectify undistorted Jingdong pinhole stereo video."""

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


def calibration(source, table):
    rows = [json.loads(line) for line in (source / "meta/calibration.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 1 or rows[0]["episode_index"] != 0:
        raise ValueError("Expected one calibration for episode zero")
    raw = rows[0]
    matrices = []
    for side in (1, 2):
        camera = raw[f"CameraParameters{side}"]
        if camera["ImageSize"] != [2160, 2160]:
            raise ValueError("Expected 2160 x 2160 calibration")
        fx, fy = camera["FocalLength"]
        cx, cy = camera["PrincipalPoint"]
        matrices.append(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]], dtype=float))
    rotation = np.asarray(raw["PoseCamera2"]["R"], dtype=float)
    translation = np.asarray(raw["PoseCamera2"]["Translation"], dtype=float) / 1000
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5):
        raise ValueError("Invalid stereo rotation")
    # Verify the provider convention against all available hand projections.
    errors = []
    for hand in ("left", "right"):
        xyz = np.asarray(table[f"{hand}_kp3d"].to_pylist()).reshape(-1, 3)
        uv = np.asarray(table[f"rightcam_{hand}_kp2d"].to_pylist()).reshape(-1, 2)
        right = xyz @ rotation.T + translation
        projected = right @ matrices[1].T
        projected = projected[:, :2] / projected[:, 2:]
        valid = np.isfinite(projected).all(1) & np.isfinite(uv).all(1) & (right[:, 2] > 0)
        errors.extend(np.linalg.norm(projected[valid] - uv[valid], axis=1))
    if not errors or np.percentile(errors, 95) > 2:
        raise ValueError("Stereo extrinsic convention failed source keypoint projection check")
    r1, r2, p1, p2, q, _, _ = cv2.stereoRectify(
        *[matrices[0], np.zeros(5), matrices[1], np.zeros(5), (2160, 2160),
          rotation, translation.reshape(3, 1)], flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
        newImageSize=(2160, 2160))
    if p2[0, 3] >= 0 or abs(p2[1, 3]) > 1e-6:
        raise ValueError("Expected horizontal left-reference stereo")
    maps = [cv2.initUndistortRectifyMap(k, np.zeros(5), r, p, (2160, 2160),
                                      cv2.CV_32FC1)
            for k, r, p in zip(matrices, (r1, r2), (p1, p2))]
    return dict(R1=r1, R2=r2, P1=p1, P2=p2, Q=q, maps=maps,
                baseline_m=float(-p2[0, 3] / p2[0, 0]),
                source_projection_p95_px=float(np.percentile(errors, 95)))


def rectify_video(source, target, maps, count, fps, ffmpeg):
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {source}")
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "2160x2160",
               "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
               "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(target)]
    log = target.with_suffix(".encode.log")
    try:
        with log.open("wb") as errors:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors)
            try:
                frames = 0
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if frame.shape[:2] != (2160, 2160):
                        raise ValueError("Unexpected source video resolution")
                    process.stdin.write(cv2.remap(frame, *maps, cv2.INTER_LINEAR).tobytes())
                    frames += 1
                process.stdin.close()
                if process.wait() != 0:
                    raise RuntimeError(f"Video encoder failed: {log.read_text(errors='replace')}")
                if frames != count:
                    raise ValueError(f"Video frame count {frames} differs from data {count}")
            except BaseException:
                process.kill()
                process.wait()
                raise
    finally:
        capture.release()
    log.unlink()
