"""Convert one Archebase session (data.mcap + annotation.json) to MF LeRobot v3.1.

ROS installation and torch are not required. Videos are remuxed without encoding;
sensor samples retain their native timestamps and source coordinate conventions.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
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
from format_convert.field_naming import HAND_LANDMARK_NAMES
from format_convert.archebase.annotations import prepare_annotation

_spec = importlib.util.spec_from_file_location("archebase_layout", REPO / "src/mf_lerobot/utils.py")
layout = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(layout)

CAMERAS = {"/decxin/left_rgb/h264": "head_left", "/decxin/right_rgb/h264": "head_right"}
CAMERA_INFO = {"/ab/cam_0/rgb/camera_info": "head_left", "/ab/cam_1/rgb/camera_info": "head_right"}
HANDS = {"/ab/pose/Left_Hand": "left", "/ab/pose/Right_Hand": "right"}
HANDS_2D = {"/ab/pose/Left_Hand_2D": "left", "/ab/pose/Right_Hand_2D": "right"}


def plain(value):
    if is_dataclass(value):
        return plain(asdict(value))
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items() if k != "__msgtype__"}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def timestamp_ns(value):
    stamp = value.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def pose_vector(pose):
    return [getattr(pose.position, axis) for axis in ("x", "y", "z")] + [
        getattr(pose.orientation, axis) for axis in ("x", "y", "z", "w")]


def increasing(times, name):
    if not times or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Missing/duplicate/non-increasing timestamps: " + name)


def read_mcap(source):
    from mcap.reader import make_reader
    from rosbags.typesys import Stores, get_typestore, get_types_from_msg
    from google.protobuf.timestamp_pb2 import Timestamp

    store, decoder = get_typestore(Stores.EMPTY), DynamicProtobufDecoder()
    streams, cameras = defaultdict(list), defaultdict(list)
    specs, calibrations, serials, counts, unknown = {}, defaultdict(list), [], Counter(), Counter()

    def add(key, t, values, names, topic, frame_id, dtype="float32", **extra):
        if len(values) != len(names):
            raise ValueError("Unexpected feature dimension: " + key)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite source values: {topic} at {t}")
        spec = dict(dtype=dtype, shape=[len(names)], names=names, source_topic=topic,
                    source_frame_id=frame_id, **extra)
        if key in specs and specs[key] != spec:
            raise ValueError("Feature layout changed within recording: " + key)
        specs[key] = spec
        streams[key].append((t, values))

    with source.open("rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()
        if summary is None:
            raise ValueError("MCAP summary is required")
        for schema in summary.schemas.values():
            if schema.encoding == "ros2msg":
                store.register(get_types_from_msg(schema.data.decode("utf-8"), schema.name))
        for schema, channel, msg in reader.iter_messages():
            topic = channel.topic
            counts[topic] += 1
            if topic in CAMERAS:
                value = decoder.decode(schema, msg.data)
                if not value or value.get("format") != "h264":
                    raise ValueError("Expected Protobuf H264 video: " + topic)
                stamp = Timestamp()
                stamp.FromJsonString(value["timestamp"])
                cameras[CAMERAS[topic]].append((stamp.ToNanoseconds(), base64.b64decode(value["data"])))
                continue
            if topic not in {*CAMERA_INFO, *HANDS, *HANDS_2D, "/decxin/imu", "/ab/slam/pose", "/decxin/serial_number"}:
                unknown[topic] += 1
                continue
            if channel.message_encoding != "cdr":
                raise ValueError("Expected CDR encoding: " + topic)
            value = store.deserialize_cdr(msg.data, schema.name)
            if topic == "/decxin/serial_number":
                serials.append(dict(log_time_ns=msg.log_time, value=value.data))
                continue
            t, frame = timestamp_ns(value), value.header.frame_id
            if topic in CAMERA_INFO:
                cal = plain(value)
                cal.pop("header")
                cal["frame_id"] = frame
                entries = calibrations[CAMERA_INFO[topic]]
                if not entries or entries[-1]["camera_info"] != cal:
                    entries.append(dict(timestamp_ns=t, camera_info=cal))
            elif topic in HANDS:
                side = HANDS[topic]
                if len(value.poses) != 21:
                    raise ValueError(f"Expected 21 {side} hand poses, got {len(value.poses)}")
                poses = np.asarray([pose_vector(p) for p in value.poses])
                for suffix, data, axes in (("points", poses[:, :3], ("x", "y", "z")),
                                           ("quaternions", poses[:, 3:], ("qx", "qy", "qz", "qw"))):
                    add(f"observation.hand_{side}_{suffix}", t, data.ravel().tolist(),
                        [f"{name}_{a}" for name in HAND_LANDMARK_NAMES for a in axes], topic, frame,
                        joint_order=HAND_LANDMARK_NAMES,
                        mapping_basis="source order confirmed by user after visualization",
                        coordinate_policy="source values unchanged; reference frame unverified")
            elif topic in HANDS_2D:
                side = HANDS_2D[topic]
                meta = dict(image_topic=value.image_topic, image_width=int(value.image_width),
                            image_height=int(value.image_height))
                prefix = f"observation.hand_{side}_"
                add(prefix+"points_2d", t, value.keypoints_uv.tolist(),
                    [f"{name}_{a}" for name in HAND_LANDMARK_NAMES for a in ("u", "v")], topic, frame, **meta)
                add(prefix+"confidence", t, [float(value.confidence)], ["confidence"], topic, frame)
            elif topic == "/ab/slam/pose":
                add("observation.head_pose", t, pose_vector(value.pose),
                    ["x", "y", "z", "qx", "qy", "qz", "qw"], topic, frame,
                    pose_semantics="source SLAM pose; tracked body/extrinsic unverified")
            elif topic == "/decxin/imu":
                add("observation.imu", t,
                    [getattr(v, a) for v in (value.linear_acceleration, value.angular_velocity) for a in ("x", "y", "z")],
                    ["ax", "ay", "az", "gx", "gy", "gz"], topic, frame,
                    units=["m/s^2"]*3+["rad/s"]*3)
    for camera in CAMERAS.values():
        if not cameras[camera] or not calibrations[camera]:
            raise ValueError("Missing video/calibration: " + camera)
    for key, samples in {**streams, **cameras}.items():
        increasing([x[0] for x in samples], key)
    return dict(streams=streams, cameras=cameras, specs=specs, calibrations=calibrations,
                serials=serials, counts=dict(counts), unknown_topics=dict(unknown))


def write_feature(root, key, samples, origin, spec=None):
    columns = dict(timestamp=pa.array([(t-origin)/1e9 for t, _ in samples], type=pa.float64()),
                   episode_index=pa.array([0]*len(samples), type=pa.int64()))
    if spec:
        dtype = {"float32": pa.float32(), "bool": pa.bool_(), "int64": pa.int64()}[spec["dtype"]]
        for i, name in enumerate(spec["names"]):
            columns[name] = pa.array([v[i] for _, v in samples], type=dtype)
    path = root / f"data/chunk-000/episode_000000/{key}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), path, compression="snappy")


def remux(samples, path, fps, ffmpeg):
    import av
    path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-f", "h264", "-r", str(fps),
                             "-i", "pipe:0", "-an", "-c:v", "copy", "-movflags", "+faststart", str(path)],
                            input=b"".join(p for _, p in samples), capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace"))
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        # Read container/codec metadata only. Video decoding belongs to later QC.
        return dict(width=stream.width, height=stream.height, codec=stream.codec_context.name,
                    pix_fmt=stream.codec_context.pix_fmt, source_packets=len(samples), fps=fps,
                    decode_validation_performed=False)


def convert(source, out, ffmpeg, fps=30.0, task=None):
    source, out = Path(source).resolve(), Path(out).resolve()
    if source.is_dir():
        source = source/"data.mcap"
    if not source.is_file():
        raise FileNotFoundError(source)
    annotation_path = source.with_name("annotation.json")
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite: {out}")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("FPS must be positive and finite")
    print(f"Read {source}", flush=True)
    records = read_mcap(source)
    master = [t for t, _ in records["cameras"]["head_left"]]
    origin = master[0]
    prepared_annotation = prepare_annotation(annotation,origin,out,source_bytes=annotation_path.read_bytes())
    task = task or annotation.get("annoDetail", {}).get("desc") or source.parent.name
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = out.with_name("."+out.name+".partial-"+uuid4().hex)
    staging.mkdir()
    try:
        features, stats, videos = {}, {}, {}
        episode = dict(episode_index=0, tasks=[task], length=len(master))
        for camera, samples in records["cameras"].items():
            key = "observation.images."+camera
            print(f"Remux {camera}: {len(samples)} packets", flush=True)
            video = remux(samples, staging/f"videos/chunk-000/{key}/episode_000000.mp4", fps, ffmpeg)
            videos[key] = video
            cal = records["calibrations"][camera][0]["camera_info"]
            if (video["width"], video["height"]) != (cal["width"], cal["height"]):
                raise ValueError("Calibration/video resolution mismatch: " + camera)
            features[key] = dict(dtype="video", shape=[video["height"],video["width"],3],
                                 names=["h","w","c"], fps=fps, tolerance_s=0.02,
                                 info={"video.fps":fps,"video.codec":video["codec"],
                                       "video.height":video["height"],"video.width":video["width"],
                                       "video.channels":3,"video.pix_fmt":video["pix_fmt"],
                                       "video.is_depth_map":False,"has_audio":False})
            write_feature(staging, key, samples, origin)
        for key, samples in records["streams"].items():
            spec = dict(records["specs"][key])
            spec["fps"] = (len(samples)-1)*1e9/(samples[-1][0]-samples[0][0]) if len(samples)>1 else 0.0
            features[key] = spec
            write_feature(staging, key, samples, origin, spec)
            a = np.asarray([v for _, v in samples], dtype=np.float64)
            stats[key] = dict(min=a.min(0).tolist(), max=a.max(0).tolist(), mean=a.mean(0).tolist(),
                              std=a.std(0).tolist(), count=[len(samples)])
            episode.update({layout.feature_length_key(key):len(samples),
                            layout.feature_from_ts_key(key):(samples[0][0]-origin)/1e9,
                            layout.feature_to_ts_key(key):(samples[-1][0]-origin)/1e9})
        path = staging/"data/chunk-000/episode_000000/master_index.parquet"
        pq.write_table(pa.table(dict(timestamp=pa.array([(t-origin)/1e9 for t in master],type=pa.float64()),
                                     frame_index=pa.array(range(len(master)),type=pa.int64()),
                                     episode_index=pa.array([0]*len(master),type=pa.int64()),
                                     index=pa.array(range(len(master)),type=pa.int64()),
                                     task_index=pa.array([0]*len(master),type=pa.int64()))), path)
        info = layout.create_empty_dataset_info(fps,{**features,**layout.DEFAULT_FEATURES},
                    master_feature="observation.images.head_left",robot_type="human_ego")
        info.update(total_episodes=1,total_frames=len(master),total_tasks=1,total_videos=len(videos),splits={"train":"0:1"})
        write_json(staging/"meta/info.json", info)
        for name, value in (("episodes",episode),("tasks",dict(task_index=0,task=task)),
                             ("episodes_stats",dict(episode_index=0,stats=stats))):
            (staging/f"meta/{name}.jsonl").write_text(json.dumps(value,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
        write_json(staging/"meta/stats.json", stats)
        shutil.copyfile(annotation_path, staging/"annotation.json")
        (staging/"annotation").mkdir()
        shutil.copyfile(annotation_path, staging/"annotation/annotation.json")
        write_json(staging/"annotation/episode_000000_source.json", prepared_annotation)
        write_json(staging/"calibrations/camera_info.json", records["calibrations"])
        write_json(staging/"meta/archebase_source.json",dict(
            source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            annotation_sha256=hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
            timestamp_origin_ns=origin,topic_counts=records["counts"],unknown_topics=records["unknown_topics"],
            serial_numbers=records["serials"],videos=videos,
            timestamp_policy="native integer nanoseconds minus first left-camera timestamp; no resampling or clipping",
            video_timestamp_policy="nominal CFR MP4; exact source timestamps in per-camera Parquet",
            nominal_video_fps=fps,nominal_video_fps_basis="CLI --fps (default 30); no source nominal FPS field",
            hand_joint_order=HAND_LANDMARK_NAMES,
            hand_mapping_basis="source order confirmed by user after visualization",
            coordinates="source values unchanged; no inferred hand/world transform",
            calibration_policy="source CameraInfo K/D/R/P retained; no inferred extrinsic or rectification"))
        staging.rename(out)
        return dict(output=str(out),frames=len(master),features=len(features),
                    stream_samples={k:len(v) for k,v in records["streams"].items()})
    except BaseException:
        print(f"Incomplete output retained: {staging}",file=sys.stderr)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",required=True,type=Path,help="Session directory or data.mcap")
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--fps",type=float,default=30.0,help="Nominal MP4 FPS; native timestamps are preserved separately")
    parser.add_argument("--task",help="Task name; defaults to annoDetail.desc or session directory name")
    parser.add_argument("--ffmpeg",help="FFmpeg executable; falls back to imageio-ffmpeg if not on PATH")
    args = parser.parse_args(argv)
    ffmpeg = args.ffmpeg or shutil.which("ffmpeg")
    if ffmpeg is None:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    result = convert(args.source,args.out,ffmpeg,args.fps,args.task)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
