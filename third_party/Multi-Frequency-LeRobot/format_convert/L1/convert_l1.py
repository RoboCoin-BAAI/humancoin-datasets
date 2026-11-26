"""Convert one L1 session, preserving each sensor's acquisition timestamps."""
from __future__ import annotations

import argparse
import csv
import hashlib
from fractions import Fraction
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from format_convert.field_naming import image_feature, MOTION_FEATURES

CAMERAS = {
    "left": "head_left", "right": "head_right",
    "bleft": "head_left_outer0", "bright": "head_right_outer0",
}
IMU_COLUMNS = [f"linear_acceleration_{axis}" for axis in "xyz"] + [
    f"angular_velocity_{axis}" for axis in "xyz"
]


def discover_sessions(source: Path) -> list[Path]:
    candidates = [source / "meta.json"] if (source / "meta.json").is_file() else sorted(source.rglob("meta.json"))
    return [p.parent for p in candidates if (p.parent / "videos").is_dir()
            and (p.parent / "timestamps").is_dir()]


def validate_session_root(source: Path) -> None:
    """Require the caller to select the actual session, without implicit descent."""
    missing = []
    if not (source / 'meta.json').is_file():
        missing.append('meta.json (file)')
    for name in ('videos', 'timestamps', 'imu'):
        if not (source / name).is_dir():
            missing.append(f'{name}/ (directory)')
    if missing:
        candidates = discover_sessions(source) if source.is_dir() else []
        hint = ('; discovered session paths: ' + ', '.join(str(p) for p in candidates)
                if candidates else '')
        raise ValueError(f'Invalid session root: {source}; missing: {", ".join(missing)}'
                         f'{hint}; pass the actual session directory explicitly '
                         '(use batch_convert.py for a batch root)')


def validate_times(times: list[float], label: str) -> None:
    if not times or any(not math.isfinite(t) for t in times):
        raise ValueError(f"{label}: empty or non-finite timestamps")
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError(f"{label}: timestamps must increase strictly")


def read_timestamps(path: Path) -> list[float]:
    times = []
    for row in path.read_text(encoding="utf-8-sig").splitlines():
        if not row.strip():
            continue
        index, timestamp = row.split()
        if int(index) != len(times):
            raise ValueError(f"{path}: non-contiguous frame indices")
        times.append(float(timestamp))
    validate_times(times, str(path))
    return times


def normalize_imu(samples):
    """Sort complete samples and retain the first acquisition at each timestamp."""
    if not samples or any(not math.isfinite(t) for t, _ in samples):
        raise ValueError("IMU: empty or non-finite timestamps")
    if any(not math.isfinite(v) for _, values in samples for v in values):
        raise ValueError("Non-finite IMU values")
    backwards = sum(b[0] < a[0] for a, b in zip(samples, samples[1:]))
    max_backwards = max((a[0] - b[0] for a, b in zip(samples, samples[1:])), default=0.)
    ordered = sorted(samples, key=lambda sample: sample[0])
    cleaned = []
    for sample in ordered:
        if not cleaned or sample[0] != cleaned[-1][0]:
            cleaned.append(sample)
    report = dict(policy="stable_sort_keep_first", input_samples=len(samples),
                  output_samples=len(cleaned), duplicates_removed=len(samples) - len(cleaned),
                  backward_steps=backwards, max_backward_s=max(0., max_backwards))
    if backwards or report["duplicates_removed"]:
        print("[warning] IMU timestamps normalized: " + json.dumps(report), file=sys.stderr, flush=True)
    return cleaned, report


def inspect_session(source: Path) -> dict:
    meta = json.loads((source / "meta.json").read_text(encoding="utf-8-sig"))
    cameras = {}
    for raw in CAMERAS:
        video = source / "videos" / f"{raw}.mkv"
        if not video.is_file() or not video.stat().st_size:
            raise ValueError(f"Missing or empty video: {video}")
        times = read_timestamps(source / "timestamps" / f"{raw}_timestamps.txt")
        expected = meta.get("topics", {}).get(raw, {}).get("written_frames")
        if expected is not None and expected != len(times):
            raise ValueError(f"{raw}: metadata has {expected} frames, timestamps have {len(times)}")
        cameras[raw] = times
    if len({len(t) for t in cameras.values()}) != 1:
        raise ValueError("Four cameras must have equal frame counts; no implicit padding")
    with (source / "imu/imu0.csv").open(encoding="utf-8-sig", newline="") as f:
        samples = [(float(row["timestamp"]), [float(row[k]) for k in IMU_COLUMNS])
                   for row in csv.DictReader(f)]
    expected = meta.get("imu", {}).get("written_samples")
    if expected is not None and expected != len(samples):
        raise ValueError("IMU sample count disagrees with metadata")
    samples, imu_normalization = normalize_imu(samples)
    fps = float(meta.get("fps", 30))
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid FPS")
    t0 = min(*(t[0] for t in cameras.values()), samples[0][0])
    return dict(meta=meta, cameras=cameras, imu=samples, t0=t0, fps=fps,
                imu_normalization=imu_normalization)


def session_duration(source: Path) -> float:
    meta = json.loads((source / 'meta.json').read_text(encoding='utf-8-sig'))
    fps = float(meta.get('fps', 30))
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid FPS')
    times = read_timestamps(source / 'timestamps/left_timestamps.txt')
    return times[-1] - times[0] + 1 / fps


def duration_allowed(duration: float) -> bool:
    return 5.0 <= duration <= 600.0


def summary(source: Path, records: dict) -> dict:
    return dict(source=str(source), time_origin_unix_s=records["t0"], fps=records["fps"],
                cameras={image_feature(CAMERAS[k]): dict(frames=len(t),
                         start_s=t[0] - records["t0"], end_s=t[-1] - records["t0"])
                         for k, t in records["cameras"].items()},
                imu_samples=len(records["imu"]),
                imu_normalization=records.get("imu_normalization"))


def probe_video(path: Path, ffprobe: str) -> dict:
    result = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0",
                             "-count_frames", "-show_entries",
                             "stream=codec_name,width,height,nb_read_frames,avg_frame_rate,start_time", "-of", "json", str(path)],
                            check=True, capture_output=True, text=True)
    streams = json.loads(result.stdout)["streams"]
    if len(streams) != 1:
        raise ValueError(f"Expected one video stream: {path}")
    info = streams[0]
    info['_decoder_errors'] = result.stderr.strip()
    if int(info.get("nb_read_frames", 0)) <= 0:
        raise ValueError(f"No decodable frames: {path}")
    return info


def write_sidecars(source: Path, out: Path, records: dict) -> None:
    audio = source / "audio/audio.wav"
    if audio.is_file():
        (out / "audio").mkdir()
        shutil.copy2(audio, out / "audio/audio.wav")
    annotation_out = out / "annotation"
    annotation_files = [
        *source.glob("*_config.yaml"),
        source / "任务详情.txt",
        source / "encoder.log",
    ]
    for path in annotation_files:
        if path.is_file():
            annotation_out.mkdir(exist_ok=True)
            shutil.copy2(path, annotation_out / path.name)
    calibration_out = out / "calibrations"
    calibration_out.mkdir()
    for path in (source / "calibrations").iterdir():
        if path.is_file():
            shutil.copy2(path, calibration_out / path.name)
    camera_mapping = {
        raw: {
            "feature_key": image_feature(canonical),
            "calibration_topic": f"/fisheye/{raw}/image_raw/compressed",
        }
        for raw, canonical in CAMERAS.items()
    }
    (calibration_out / "lerobot_camera_mapping.json").write_text(
        json.dumps(camera_mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    annotation = summary(source, records)
    annotation.update(audio_synchronization="unknown", task=records["task"])
    (out / "meta/l1_source.json").write_text(json.dumps(annotation, ensure_ascii=False, indent=2), encoding="utf-8")


def convert(source: Path, out: Path, records: dict, ffmpeg: str, ffprobe: str) -> None:
    for executable in (ffmpeg, ffprobe):
        if shutil.which(executable) is None:
            raise ValueError(f"Required executable not found: {executable}")
    # Import the heavy dataset stack only for a real conversion.
    import numpy as np
    import yaml
    from mf_lerobot import MultiFrequencyLeRobotDataset

    if out.exists():
        raise ValueError(f"Output already exists: {out}")
    if out == source or out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError("Source and output directories must not overlap")
    video_info = {}
    for raw, times in records["cameras"].items():
        info = probe_video(source / "videos" / f"{raw}.mkv", ffprobe)
        if int(info["nb_read_frames"]) != len(times):
            raise ValueError(f"{raw}: decoded video count differs from timestamps")
        if info["codec_name"] != "h264":
            raise ValueError(f"{raw}: expected H264 for lossless MP4 remux")
        video_info[raw] = info
    tasks = []
    for path in sorted(source.glob("*_config.yaml")):
        config = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
        if config.get("task"):
            tasks.append(str(config["task"]))
    if len(set(tasks)) > 1:
        raise ValueError("Conflicting task labels")
    records["task"] = tasks[0] if tasks else str(records["meta"].get("task", source.name))
    specs = {image_feature(CAMERAS[raw]): dict(dtype="video", shape=(v["height"], v["width"], 3),
             names=["h", "w", "c"], fps=records["fps"], tolerance_s=0.02)
             for raw, v in video_info.items()}
    imu = records["imu"]
    rate = (len(imu) - 1) / (imu[-1][0] - imu[0][0]) if len(imu) > 1 else 200.0
    specs[MOTION_FEATURES["imu"]] = dict(dtype="float32", shape=(6,),
        names=["ax", "ay", "az", "gx", "gy", "gz"], fps=rate, window=(-1 / records["fps"], 0.0))
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = out.parent / f".{out.name}.partial-{uuid4().hex}"
    try:
        ds = MultiFrequencyLeRobotDataset.create(repo_id=out.name, root=staging,
            fps=records["fps"], features=specs, master_feature=image_feature(CAMERAS["left"]))
        for t in records["cameras"]["left"]:
            ds.add_frame("task", records["task"], t - records["t0"])
        for t, values in imu:
            ds.add_frame(MOTION_FEATURES["imu"], np.asarray(values, dtype=np.float32), t - records["t0"])
        validated_videos = {}
        for raw, times in records["cameras"].items():
            path = ds._features[image_feature(CAMERAS[raw])].add_external_video([t - records["t0"] for t in times])
            path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-i", str(source / "videos" / f"{raw}.mkv"),
                            "-map", "0:v:0", "-an", "-c:v", "copy", "-movflags", "+faststart", str(path)], check=True)
            exported = probe_video(path, ffprobe)
            if int(exported["nb_read_frames"]) != len(times):
                raise ValueError(f"{raw}: remux changed frame count")
            # VideoFeature.read decodes frame k at k/fps. Reject a container
            # time base that would make that indexing select the wrong frame.
            actual_fps = float(Fraction(exported.get("avg_frame_rate", "0/1")))
            if abs(actual_fps - records["fps"]) > 0.001 * records["fps"]:
                raise ValueError(f"{raw}: MP4 FPS {actual_fps} differs from declared {records['fps']}")
            if abs(float(exported.get("start_time", "0"))) > 0.5 / records["fps"]:
                raise ValueError(f"{raw}: MP4 video timeline does not start at zero")
            validated_videos[raw] = (path, exported)
        ds.save_episode()
        write_sidecars(source, staging, records)
        receipt = dict(version=1, producer='l1_converter', videos={})
        for raw, (video_path, stream) in validated_videos.items():
            tables = list(staging.glob(f'data/chunk-*/episode_*/{image_feature(CAMERAS[raw])}.parquet'))
            if len(tables) != 1 or stream.get('_decoder_errors', ''):
                continue
            stat = video_path.stat()
            receipt['videos'][video_path.relative_to(staging).as_posix()] = dict(
                identity=dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                              timestamps_sha256=hashlib.sha256(tables[0].read_bytes()).hexdigest()),
                stream={k: v for k,v in stream.items() if not k.startswith('_')}, decoder_errors='')
        (staging / 'meta/l1_video_validation.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
        staging.rename(out)
    except BaseException:
        print(f"Incomplete output retained for inspection: {staging}", file=sys.stderr)
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Validate metadata/timestamps/IMU without video dependencies")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args(argv)
    try:
        source = args.source.resolve()
        validate_session_root(source)
        duration = session_duration(source)
        if not duration_allowed(duration):
            print(json.dumps(dict(status='skipped_duration', source=str(source), duration_s=duration,
                                  min_s=5., max_s=600.)))
            return 0
        records = inspect_session(source)
        print(json.dumps(summary(source, records), ensure_ascii=False, indent=2))
        if not args.dry_run:
            convert(source, args.out.resolve(), records, args.ffmpeg, args.ffprobe)
            print(f"[done] {args.out}")
        return 0
    except (OSError, ValueError, KeyError, ImportError, subprocess.CalledProcessError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
