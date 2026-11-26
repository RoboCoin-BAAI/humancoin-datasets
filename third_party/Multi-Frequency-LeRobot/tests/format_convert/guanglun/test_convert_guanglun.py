import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from format_convert.guanglun import convert_guanglun_mcap as converter


def test_nanosecond_time_and_pose_order():
    epoch = 1785145575556999936
    assert converter.timestamp_ns({"header": {"ts": str(epoch)}}, 0) == epoch
    assert converter.timestamp_ns({"timestamp": "2026-07-27T09:46:15.556999936Z"}, 0) == epoch
    assert converter.timestamp_ns({}, 123) == 123
    assert converter.pose_vector({"transforms": [{"pos": {"x": 2}, "quat": {"w": 1}}]}) == [2, 0, 0, 0, 0, 0, 1]
    with pytest.raises(ValueError):
        converter.strictly_increasing([2, 2], "camera")


def test_hand_xyz_and_quaternions_are_split_without_reordering():
    sample = dict(transforms=[dict(pos=dict(x=i, y=i+1, z=i+2),
                                    quat=dict(x=.1, y=.2, z=.3, w=.4)) for i in range(21)])
    xyz, quaternion = converter.split_hand(sample)
    assert len(xyz) == 63 and len(quaternion) == 84
    assert xyz[:6] == [0, 1, 2, 1, 2, 3]
    assert quaternion[:8] == [.1, .2, .3, .4, .1, .2, .3, .4]
    assert converter.hand_spec("observation.hand_left_points")["names"][:3] == ["wrist_x", "wrist_y", "wrist_z"]
    assert converter.hand_spec("observation.hand_right_quaternions")["names"][-1] == "pinkyTip_qw"
    with pytest.raises(ValueError, match="21 hand"):
        converter.split_hand(dict(transforms=sample["transforms"][:20]))


def test_native_timestamp_table(tmp_path):
    import pyarrow.parquet as pq
    origin = 1785145575556999936
    converter.write_feature(tmp_path, "observation.head_pose", [origin, origin+33333333], origin,
                            [[1, 2], [3, 4]], ["x", "y"])
    table = pq.read_table(tmp_path/"data/chunk-000/episode_000000/observation.head_pose.parquet")
    assert table["timestamp"].to_pylist() == [0, .033333333]
    assert table["x"].to_pylist() == [1, 3]


def make_sample(path):
    """Small real protobuf MCAP with two stereo H264 frames and a pose stream."""
    import av
    import numpy as np
    from fractions import Fraction
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
    from mcap.writer import Writer

    fd = descriptor_pb2.FileDescriptorProto(name="test.proto", package="test", syntax="proto3")
    def message(name, fields):
        m = fd.message_type.add(name=name)
        for i, (fname, kind, type_name, repeated) in enumerate(fields, 1):
            field = m.field.add(name=fname, number=i, type=kind, label=3 if repeated else 1)
            if type_name:
                field.type_name = ".test."+type_name
    message("Header", [("ts", 3, None, False)])
    message("Device", [("device_type", 9, None, False), ("fps", 9, None, False)])
    message("Task", [("task_name", 9, None, False)])
    message("Metadata", [("header", 11, "Header", False), ("devices", 11, "Device", True),
                         ("task_info", 11, "Task", False)])
    message("Video", [("header", 11, "Header", False), ("format", 9, None, False), ("data", 12, None, False)])
    message("Position", [("x", 1, None, False)])
    message("Quaternion", [("w", 1, None, False)])
    message("Transform", [("pos", 11, "Position", False), ("quat", 11, "Quaternion", False)])
    message("Pose", [("header", 11, "Header", False), ("transforms", 11, "Transform", True)])
    pool = descriptor_pool.DescriptorPool()
    pool.Add(fd)
    classes = {name: message_factory.GetMessageClass(pool.FindMessageTypeByName("test."+name))
               for name in ("Metadata", "Video", "Pose")}
    ds = descriptor_pb2.FileDescriptorSet()
    ds.file.add().CopyFrom(fd)
    codec = av.CodecContext.create("libx264", "w")
    codec.width, codec.height, codec.pix_fmt = 64, 48, "yuv420p"
    codec.time_base = Fraction(1, 30)
    codec.framerate = Fraction(30, 1)
    codec.options = {"preset": "ultrafast", "tune": "zerolatency", "bf": "0"}
    packets = []
    for i in range(2):
        frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), i*80, dtype=np.uint8), format="rgb24")
        frame.pts = i
        packets.extend(bytes(p) for p in codec.encode(frame))
    packets.extend(bytes(p) for p in codec.encode(None))
    assert len(packets) == 2
    with path.open("wb") as stream:
        writer = Writer(stream)
        writer.start()
        channels = {}
        for topic, name in (("/session/metadata", "Metadata"), ("/pose/head", "Pose"),
                            ("/pose/left_hand", "Pose"), ("/pose/right_hand", "Pose"),
                            ("/sensor/camera/head_left/video", "Video"),
                            ("/sensor/camera/head_right/video", "Video")):
            schema = writer.register_schema("test."+name, "protobuf", ds.SerializeToString())
            channels[topic] = writer.register_channel(topic, "protobuf", schema)
        origin = 1700000000000000000
        meta = classes["Metadata"]()
        meta.header.ts = origin
        meta.task_info.task_name = "test task"
        for side in ("left", "right"):
            meta.devices.add(device_type="head_"+side+"_camera", fps="30")
        writer.add_message(channels["/session/metadata"], origin, meta.SerializeToString(), origin)
        for i, packet in enumerate(packets):
            t = origin+i*33333333
            pose = classes["Pose"]()
            pose.header.ts = t
            transform = pose.transforms.add()
            transform.pos.x = i*.01
            transform.quat.w = 1
            writer.add_message(channels["/pose/head"], t, pose.SerializeToString(), t)
            for side in ("left", "right"):
                hand = classes["Pose"]()
                hand.header.ts = t
                for j in range(21):
                    joint = hand.transforms.add()
                    joint.pos.x = j*.01+i*.01
                    joint.quat.w = 1
                writer.add_message(channels["/pose/"+side+"_hand"], t, hand.SerializeToString(), t)
                video = classes["Video"](format="h264", data=packet)
                video.header.ts = t
                writer.add_message(channels["/sensor/camera/head_"+side+"/video"], t,
                                   video.SerializeToString(), t)
        writer.finish()


def test_end_to_end_remux_preserves_episode_and_refuses_overwrite(tmp_path):
    import imageio_ffmpeg
    import pyarrow.parquet as pq
    source, out = tmp_path/"sample.mcap", tmp_path/"converted"
    make_sample(source)
    result = converter.convert(source, out, imageio_ffmpeg.get_ffmpeg_exe())
    assert result["frames"] == 2
    assert not (out/"source_records").exists()
    info = json.loads((out/"meta/info.json").read_text())
    assert info["features"]["observation.head_pose"]["shape"] == [7]
    assert info["features"]["observation.hand_left_points"]["shape"] == [63]
    source_info = json.loads((out/"meta/guanglun_source.json").read_text())
    assert source_info["rectification_status"] == "native_rectified"
    table = pq.read_table(out/"data/chunk-000/episode_000000/observation.head_pose.parquet")
    assert table["x"].to_pylist() == pytest.approx([0, .01])
    assert table["timestamp"].to_pylist() == [0, .033333333]
    assert len(list((out/"videos").glob("*/*/*.mp4"))) == 2
    assert info["master_feature"] == "observation.images.head_left_rectified"
    assert "observation.images.head_left" not in info["features"]
    assert not source_info["rectification_performed_by_converter"]
    assert (out/"data/chunk-000/episode_000000/observation.images.head_right_rectified.parquet").is_file()
    calibration = json.loads((out/"calibrations/source_camera_calibration.json").read_text())
    assert calibration["head_left_rectified"]["raw_camera_id"] == "head_left"
    assert calibration["head_left_rectified"]["feature_key"] == "observation.images.head_left_rectified"
    assert {p.name for p in (out/"meta").iterdir()} == {
        "info.json", "episodes.jsonl", "tasks.jsonl", "episodes_stats.jsonl",
        "guanglun_source.json", "guanglun_video_validation.json",
    }
    episode_stats = json.loads((out/"meta/episodes_stats.jsonl").read_text())
    assert episode_stats["episode_index"] == 0
    assert episode_stats["stats"]["observation.head_pose"]["count"] == [2]
    assert info["features"]["observation.images.head_left_rectified"]["names"] == ["h", "w", "c"]
    assert info["features"]["observation.images.head_left_rectified"]["tolerance_s"] == .02
    master = pq.read_table(out/"data/chunk-000/episode_000000/master_index.parquet")
    assert str(master.schema.field("timestamp").type) == "float"
    with pytest.raises(FileExistsError):
        converter.convert(source, out, imageio_ffmpeg.get_ffmpeg_exe())
