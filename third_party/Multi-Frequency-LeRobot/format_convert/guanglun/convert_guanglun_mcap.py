"""Convert one Lightwheel EgoStandard MCAP to the repository's v3.1 layout.

No training/GPU dependency is needed. H264/HEVC packets are remuxed, not encoded.
Pose transform order and coordinate frames are preserved, never guessed.
"""
from __future__ import annotations

import argparse
import base64
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from uuid import uuid4

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from format_convert.jianzhi.convert_jianzhi_mcap import DynamicProtobufDecoder
from format_convert.guanglun.format_semantics import format_segments
from format_convert.guanglun.calibration import compact as compact_calibration, export as export_calibration
from format_convert.field_naming import HAND_FEATURES, HAND_LANDMARK_NAMES
from format_convert.guanglun.pose_qc import check as check_pose_jumps, Limits, PoseJumpError

# Load dependency-free layout helpers without importing the torch-based package.
_spec = importlib.util.spec_from_file_location("mf_layout", REPO / "src/mf_lerobot/utils.py")
layout = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(layout)

POSES = {
    "/pose/head": "head_pose",
    "/pose/headcam": "head_camera_pose",
    "/pose/right_eye_cam": "right_eye_camera_pose",
    "/pose/left_hand": "hand_left_pose",
    "/pose/right_hand": "hand_right_pose",
    "/pose/body": "body_pose",
    "/pose/upper_body": "upper_body_pose",
    "/pose/lower_body": "lower_body_pose",
}
CAMERAS = ("head_left", "head_right")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def timestamp_ns(value, fallback):
    header = value.get("header", {})
    if "ts" in header:
        return int(header["ts"])
    if value.get("timestamp"):
        from google.protobuf.timestamp_pb2 import Timestamp
        stamp = Timestamp()
        stamp.FromJsonString(value["timestamp"])
        return stamp.ToNanoseconds()
    transforms = value.get("transforms", [])
    if transforms and transforms[0].get("timestamp"):
        return timestamp_ns(transforms[0], fallback)
    return int(fallback)


def strictly_increasing(times, topic):
    if not times or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError(f"Missing/duplicate/non-increasing timestamps: {topic}")


def pose_vector(value):
    result = []
    for transform in value.get("transforms", []):
        pos, quat = transform.get("pos", {}), transform.get("quat", {})
        result.extend(float(pos.get(k, 0)) for k in ("x", "y", "z"))
        result.extend(float(quat.get(k, 0)) for k in ("x", "y", "z", "w"))
    if not result or not np.isfinite(result).all():
        raise ValueError("Empty or non-finite pose transform array")
    return result


def split_hand(value):
    transforms = np.asarray(pose_vector(value)).reshape(-1, 7)
    if len(transforms) != 21:
        raise ValueError(f"Expected 21 hand transforms, got {len(transforms)}")
    return transforms[:, :3].ravel().tolist(), transforms[:, 3:].ravel().tolist()


def hand_spec(key):
    quaternion = key.endswith("_quaternions")
    axes = ("qx", "qy", "qz", "qw") if quaternion else ("x", "y", "z")
    result = dict(dtype="float32", shape=[21*len(axes)],
                  names=[f"{joint}_{axis}" for joint in HAND_LANDMARK_NAMES for axis in axes],
                  landmark_order=HAND_LANDMARK_NAMES,
                  mapping_basis="source order accepted after sample visualization and user confirmation")
    if quaternion:
        result.update(quaternion_order="xyzw", rotation_convention="source joint orientations, unchanged")
    else:
        result.update(coordinate_frame="world", coordinate_policy="source XYZ, no transformation")
    return result


def read_mcap(source):
    from mcap.reader import make_reader
    decoder = DynamicProtobufDecoder()
    streams, cameras, raw = defaultdict(list), defaultdict(list), defaultdict(list)
    counts = defaultdict(int)
    with source.open("rb") as f:
        for schema, channel, message in make_reader(f).iter_messages():
            value = decoder.decode(schema, message.data)
            if value is None:
                raise ValueError(f"Unsupported schema: {channel.topic}")
            topic = channel.topic
            stamp = timestamp_ns(value, message.log_time)
            counts[topic] += 1
            if topic.startswith("/sensor/camera/") and topic.endswith("/video"):
                camera = topic.split("/")[3]
                if camera not in CAMERAS:
                    raise ValueError(f"Unmapped camera: {camera}")
                fmt = value.get("format", "").lower()
                if fmt not in ("h264", "h265", "hevc"):
                    raise ValueError(f"Unsupported compressed video: {fmt}")
                cameras[camera].append((stamp, base64.b64decode(value["data"], validate=True), fmt))
            else:
                raw[topic].append(dict(timestamp_ns=stamp, log_time_ns=message.log_time,
                                       publish_time_ns=message.publish_time, data=value))
                if topic in ("/pose/left_hand", "/pose/right_hand"):
                    side = "left" if topic == "/pose/left_hand" else "right"
                    xyz, quat = split_hand(value)
                    streams[HAND_FEATURES[side+"_points"]].append((stamp, xyz))
                    streams[f"observation.hand_{side}_quaternions"].append((stamp, quat))
                elif topic in POSES:
                    streams["observation." + POSES[topic]].append((stamp, pose_vector(value)))
                elif topic.startswith("/annotation/bad_frame/"):
                    key = "observation." + topic.removeprefix("/annotation/").replace("/", "_")
                    streams[key].append((stamp, [bool(value.get("is_bad", False))]))
    for camera in CAMERAS:
        strictly_increasing([x[0] for x in cameras[camera]], camera)
        if len({x[2] for x in cameras[camera]}) != 1:
            raise ValueError(f"Video encoding changed: {camera}")
    if len(cameras["head_left"]) != len(cameras["head_right"]):
        raise ValueError("Left/right frame counts differ; no implicit padding is allowed")
    for key, samples in streams.items():
        strictly_increasing([x[0] for x in samples], key)
        if len({len(x[1]) for x in samples}) != 1:
            raise ValueError(f"Pose transform count changed: {key}")
    return dict(cameras=cameras, streams=streams, raw=raw, counts=dict(counts))


def write_feature(root, key, times_ns, origin, values=None, names=None, boolean=False):
    columns = {"timestamp": pa.array([(t-origin)/1e9 for t in times_ns], type=pa.float64()),
               "episode_index": pa.array([0]*len(times_ns), type=pa.int64())}
    if values is not None:
        for i, name in enumerate(names):
            columns[name] = pa.array([v[i] for v in values], type=pa.bool_() if boolean else pa.float32())
    path = root / f"data/chunk-000/episode_000000/{key}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), path, compression="snappy")


def remux(samples, destination, fps, ffmpeg):
    fmt = "hevc" if samples[0][2] in ("hevc", "h265") else "h264"
    destination.parent.mkdir(parents=True, exist_ok=True)
    run = subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-f", fmt,
                          "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "copy",
                          "-movflags", "+faststart", str(destination)],
                         input=b"".join(x[1] for x in samples), capture_output=True)
    if run.returncode:
        raise RuntimeError(run.stderr.decode(errors="replace"))
    import av
    with av.open(str(destination)) as container:
        stream = container.streams.video[0]
        pts = [frame.time for frame in container.decode(video=0)]
        if stream.codec_context.has_b_frames:
            raise ValueError("B-frame input requires explicit packet/display timestamp mapping; refusing implicit alignment")
        # A one-time sample conversion check; no image files or re-encoding.
        if len(pts) != len(samples):
            raise ValueError(f"Packet/decoded frame mismatch: {len(samples)} vs {len(pts)}")
        if any(t is None for t in pts) or any(b <= a for a,b in zip(pts,pts[1:])):
            raise ValueError("Invalid MP4 presentation timeline")
        if any(abs(t-i/fps) > .5/fps for i,t in enumerate(pts)):
            raise ValueError("MP4 frame indexing does not match nominal FPS")
        return dict(width=stream.width, height=stream.height,
                    codec=stream.codec_context.name, fps=float(stream.average_rate),
                    decoded_frames=len(pts), pix_fmt=stream.codec_context.pix_fmt,
                    has_b_frames=bool(stream.codec_context.has_b_frames))


def convert(source, out, ffmpeg, pose_limits=None):
    source, out = source.resolve(), out.resolve()
    if not source.is_file() or source.suffix.lower() != ".mcap":
        raise ValueError("Source must be one MCAP file")
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite: {out}")
    records = read_mcap(source)
    pose_qc = check_pose_jumps(records, pose_limits)
    if not pose_qc["passed"]:
        raise PoseJumpError(pose_qc)
    print("Head/hand pose jump check passed",flush=True)
    metadata_rows = records["raw"].get("/session/metadata", [])
    if len(metadata_rows) != 1:
        raise ValueError("Expected exactly one session metadata message")
    metadata = metadata_rows[0]["data"]
    devices = {x["device_type"]: x for x in metadata.get("devices", [])}
    task = metadata.get("task_info", {}).get("task_name", source.stem)
    master = [x[0] for x in records["cameras"]["head_left"]]
    origin = master[0]
    for camera in CAMERAS:
        if camera+"_camera" not in devices:
            raise ValueError(f"Missing device metadata: {camera}")
    fps = float(devices["head_left_camera"]["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid nominal camera FPS")
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = out.with_name("."+out.name+".partial-"+uuid4().hex)
    staging.mkdir()
    try:
        features, stats, episode, videos = {}, {}, dict(episode_index=0, tasks=[task], length=len(master)), {}
        for camera in CAMERAS:
            key = "observation.images."+camera+"_rectified"
            samples = records["cameras"][camera]
            camera_fps = float(devices[camera+"_camera"]["fps"])
            print(f"Remux {camera}: {len(samples)} packets", flush=True)
            path = staging/f"videos/chunk-000/{key}/episode_000000.mp4"
            videos[key] = remux(samples, path, camera_fps, ffmpeg)
            info = videos[key]
            features[key] = dict(dtype="video", shape=[info["height"],info["width"],3],
                                 names=["h","w","c"], fps=camera_fps, tolerance_s=0.02,
                                 info={"video.fps":camera_fps,"video.codec":info["codec"],
                                       "video.height":info["height"],"video.width":info["width"],
                                       "video.channels":3,
                                       "video.pix_fmt":info["pix_fmt"],"video.is_depth_map":False,
                                       "has_audio":False})
            write_feature(staging,key,[x[0] for x in samples],origin)
        for key, samples in records["streams"].items():
            values = [x[1] for x in samples]
            boolean = key.startswith("observation.bad_frame_")
            width = len(values[0])
            names = (["is_bad"] if boolean else
                     [f"transform_{i:03d}_{dim}" for i in range(width//7)
                      for dim in ("x","y","z","qx","qy","qz","qw")])
            if width == 7:
                names = ["x","y","z","qx","qy","qz","qw"]
            hand = key in (HAND_FEATURES["left_points"], HAND_FEATURES["right_points"],
                           "observation.hand_left_quaternions", "observation.hand_right_quaternions")
            if hand:
                names = hand_spec(key)["names"]
            rate = (len(samples)-1)*1e9/(samples[-1][0]-samples[0][0]) if len(samples)>1 else fps
            features[key] = dict(dtype="bool" if boolean else "float32",shape=[width],names=names,fps=rate)
            if hand:
                features[key].update(hand_spec(key))
            write_feature(staging,key,[x[0] for x in samples],origin,values,names,boolean)
            if not boolean:
                a = np.asarray(values, dtype=np.float32)
                stats[key] = dict(min=a.min(0).tolist(),max=a.max(0).tolist(),
                                  mean=a.mean(0).tolist(),std=a.std(0).tolist(),count=[len(samples)])
        segments = records["raw"].get("/annotation/semantic_segments", [])
        write_json(staging/"annotation/semantic_segments.json", [r["data"] for r in segments])
        write_json(staging/"annotation/episode_000000_structured.json", format_segments(
            staging/"annotation/semantic_segments.json", out/"data/chunk-000/episode_000000",
            source_reference=out/"annotation/semantic_segments.json"))
        calibration = {camera+"_rectified": dict(
            raw_camera_id=camera, feature_key="observation.images."+camera+"_rectified",
            intrinsic=records["raw"].get(f"/sensor/camera/{camera}/intrinsic", []),
            extrinsic=records["raw"].get(f"/sensor/camera/{camera}/extrinsic", [])) for camera in CAMERAS}
        # Preserve native camera-to-world poses and source frame IDs; label published features only.
        export_calibration(staging, out, compact_calibration(calibration), videos)
        index = pa.table(dict(timestamp=pa.array([(t-origin)/1e9 for t in master],type=pa.float32()),
                              frame_index=pa.array(range(len(master)),type=pa.int64()),
                              episode_index=pa.array([0]*len(master),type=pa.int64()),
                              index=pa.array(range(len(master)),type=pa.int64()),
                              task_index=pa.array([0]*len(master),type=pa.int64())))
        pq.write_table(index,staging/"data/chunk-000/episode_000000/master_index.parquet")
        info = layout.create_empty_dataset_info(fps,{**features,**layout.DEFAULT_FEATURES},
                   master_feature="observation.images.head_left_rectified",robot_type="human_ego")
        info.update(total_episodes=1,total_frames=len(master),total_tasks=1,total_videos=2,splits={"train":"0:1"})
        write_json(staging/"meta/info.json",info)
        # Match L1 and the metadata writer inherited from pinned lerobot==0.3.3.
        # The repository's v3.1 version denotes per-feature data layout; it does
        # not switch this metadata writer to upstream LeRobot v3 Parquet files.
        (staging/"meta/episodes.jsonl").write_text(json.dumps(episode)+"\n",encoding="utf-8")
        (staging/"meta/tasks.jsonl").write_text(json.dumps(dict(task_index=0,task=task))+"\n",encoding="utf-8")
        (staging/"meta/episodes_stats.jsonl").write_text(
            json.dumps(dict(episode_index=0,stats=stats),allow_nan=False)+"\n",encoding="utf-8")
        write_json(staging/"meta/guanglun_source.json",dict(
            source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            collection_metadata=metadata,
            timestamp_origin_ns=origin,topic_counts=records["counts"],videos=videos,
            pose_jump_check=pose_qc,
            source_transform_order="canonical 21-landmark order accepted after visualization and user confirmation",
            hand_position_frame="world", hand_quaternion_order="xyzw",
            pose_coordinates="source coordinates; no world/camera transform applied",
            rectification_status="native_rectified",
            rectification_basis="user-confirmed source rectification; sample epipolar alignment checked",
            rectification_performed_by_converter=False,
            video_timestamp_policy="nominal CFR MP4; per-stream timestamps preserved in Parquet"))
        validation = dict(version=1,producer="guanglun_converter",videos={})
        for key, stream in videos.items():
            path=staging/f"videos/chunk-000/{key}/episode_000000.mp4"
            stamps=staging/f"data/chunk-000/episode_000000/{key}.parquet"
            stat=path.stat()
            validation["videos"][path.relative_to(staging).as_posix()]=dict(
                identity=dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns,
                              timestamps_sha256=hashlib.sha256(stamps.read_bytes()).hexdigest()),
                stream=stream,decoder_errors="")
        write_json(staging/"meta/guanglun_video_validation.json",validation)
        staging.rename(out)
        return dict(output=str(out),frames=len(master),features=len(features),videos=videos)
    except BaseException:
        print(f"Incomplete output retained: {staging}",file=sys.stderr)
        raise


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--ffmpeg",default="ffmpeg")
    p.add_argument("--head-jump-m",type=float,default=.40)
    p.add_argument("--hand-jump-m",type=float,default=.60)
    p.add_argument("--head-jump-deg",type=float,default=120.)
    p.add_argument("--hand-jump-deg",type=float,default=160.)
    a=p.parse_args(argv)
    if not shutil.which(a.ffmpeg):
        p.error("ffmpeg not found; pass --ffmpeg PATH")
    limits = Limits(a.head_jump_m,a.hand_jump_m,a.head_jump_deg,a.hand_jump_deg)
    try:
        result = convert(a.source,a.out,a.ffmpeg,limits)
    except PoseJumpError as exc:
        print(json.dumps(dict(error=str(exc),pose_jump_check=exc.report),indent=2),file=sys.stderr)
        return 2
    print(json.dumps(result,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
