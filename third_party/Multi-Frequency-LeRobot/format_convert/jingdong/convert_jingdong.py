"""Convert a Jingdong instance or batch into per-feature HumanCoin datasets."""

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import re

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from format_convert.jingdong.stereo import calibration, rectify_video
spec = importlib.util.spec_from_file_location("mf_layout", REPO / "src/mf_lerobot/utils.py")
layout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layout)

CAMERAS = {
    "observation.images.cam_head_ego_left": "observation.images.head_left_rectified",
    "observation.images.cam_head_ego_right": "observation.images.head_right_rectified",
}
RESERVED = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
RAW_HAND_POINTS = {
    "left_kp3d": "observation.raw_hand_left_points",
    "right_kp3d": "observation.raw_hand_right_points",
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def structured_annotation(source, output, segments, timestamps, fps):
    """Extract source text and intervals using the Argus v1 schema."""
    task_rows = read_lines(source / "meta/tasks.jsonl")
    descriptions = list(dict.fromkeys(row["task_desc_en"] for row in task_rows
                                     if row.get("task_desc_en")))
    subtasks = []
    for segment in segments:
        text = segment["language_instruction"]
        words = re.findall(r"[A-Za-z]+", text)
        if not words:
            raise ValueError("Source segment has no action text")
        start, end = segment["start_frame"], segment["end_frame"]
        # Source intervals include end_frame; output intervals end at the next frame.
        end_s = float(timestamps[end + 1]) if end + 1 < len(timestamps) else float(timestamps[end]) + 1 / fps
        subtasks.append(dict(
            start_s=float(timestamps[start]), end_s=end_s, arm=None,
            action=[words[0].lower()], object=None, destination=None,
            spatial_relation=None, contribution=None, progress=None,
            hands_visible=None, hands_wearing=None, text=text))
    raw = (source / "meta/episodes.jsonl").read_bytes()
    scenes = list(dict.fromkeys(
        row[key] for row in task_rows
        for key in ("scene_cn", "first_scene_cn", "first_scene_en",
                    "second_scene_cn", "second_scene_en", "third_scene_cn", "third_scene_en")
        if row.get(key)))
    return dict(
        schema_version="1.0", episode_dir=str(output / "data/chunk-000/episode_000000"),
        source=dict(file=str(source / "meta/episodes.jsonl"),
                    sha256=hashlib.sha256(raw).hexdigest()),
        timesteps_s=sorted({time for row in subtasks for time in (row["start_s"], row["end_s"])}),
        finish_reason=None, objects=[], scene=None, subtask=subtasks,
        task_description="\n".join(descriptions) or None, task_name=None,
        key_events=[], state_changes=[], tasks=[], scene_graph=[], recovery=[],
        instruction_variants=[], data_issues=[], viewpoint=None, performance_review=None,
        actions=dict(vocabulary=sorted({a for row in subtasks for a in row["action"]})),
        scene_classification=dict(category="unknown", area="unknown",
                                  original_description=" / ".join(scenes) or None),
        annotation_provenance=dict(
            provider="jingdong", task_metadata=task_rows,
            action_extraction="first English word of each source instruction; no taxonomy inference",
            missing_evidence="arm, objects, destination, scene taxonomy and progress remain unspecified"))


def convert(source, output, ffmpeg="ffmpeg"):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Source and output directories must be disjoint")
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    info = json.loads((source / "meta/info.json").read_text(encoding="utf-8"))
    if info["total_episodes"] != 1:
        raise ValueError("Expected one episode per Jingdong instance")
    table = pq.read_table(source / info["data_path"].format(episode_chunk=0, episode_index=0))
    count = table.num_rows
    if count == 0 or count != info["total_frames"]:
        raise ValueError("Empty data or total_frames mismatch")
    timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64).reshape(count, -1)
    if timestamps.shape[1] != 1:
        raise ValueError("Expected one timestamp per row")
    timestamps = timestamps[:, 0]
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError("Timestamps must be finite and strictly increasing")
    if table["frame_index"].to_pylist() != list(range(count)):
        raise ValueError("Expected consecutive frame indices starting from zero")
    if set(table["episode_index"].to_pylist()) != {0}:
        raise ValueError("Unexpected episode indices")
    segments = read_lines(source / "meta/episodes.jsonl")
    tasks, task_ids = [], np.full(count, -1, dtype=np.int64)
    for segment in segments:
        start, end = segment["start_frame"], segment["end_frame"] + 1
        if segment["episode_index"] != 0 or not 0 <= start < end <= count:
            raise ValueError("Invalid source annotation range")
        if np.any(task_ids[start:end] >= 0):
            raise ValueError("Overlapping source annotation ranges")
        text = segment["language_instruction"]
        if text not in tasks:
            tasks.append(text)
        task_ids[start:end] = tasks.index(text)
    if np.any(task_ids < 0):
        raise ValueError("Source annotations do not cover every frame")
    stereo = calibration(source, table)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="." + output.name + "-", dir=output.parent))
    try:
        data = stage / "data/chunk-000/episode_000000"
        data.mkdir(parents=True)
        features, stats, mapping = {}, {}, {}
        common = {"timestamp": pa.array(timestamps), "episode_index": pa.array([0] * count)}
        for camera_index, (raw, key) in enumerate(CAMERAS.items()):
            original = source / info["video_path"].format(
                episode_chunk=0, episode_index=0, video_key=raw)
            target = stage / layout.DEFAULT_VIDEO_PATH.format(
                episode_chunk=0, episode_index=0, video_key=key)
            target.parent.mkdir(parents=True, exist_ok=True)
            rectify_video(original, target, stereo["maps"][camera_index], count, info["fps"], ffmpeg)
            features[key] = copy.deepcopy(info["features"][raw])
            features[key]["fps"] = info["fps"]
            features[key]["info"] = features[key].pop("video_info", {})
            features[key]["info"].update({"video.codec": "h264", "video.pix_fmt": "yuv420p"})
            pq.write_table(pa.table(common), data / (key + ".parquet"))
            mapping[raw] = key
        for raw in table.column_names:
            if raw not in {*RAW_HAND_POINTS, "camera_pos"}:
                continue
            key = raw if raw.startswith("observation.") or raw == "action" else "observation.source." + raw
            key = RAW_HAND_POINTS.get(raw, key)
            values = table[raw].to_pylist()
            try:
                array = np.asarray(values, dtype=np.float64).reshape(count, -1)
            except (ValueError, TypeError):
                # Object and occlusion structs are outside this numeric feature layout.
                continue
            if not np.isfinite(array).all():
                raise ValueError(f"Non-finite source values: {raw}")
            names = [f"value_{i:03d}" for i in range(array.shape[1])]
            columns = dict(common)
            columns.update({name: pa.array(array[:, i]) for i, name in enumerate(names)})
            pq.write_table(pa.table(columns), data / (key + ".parquet"))
            features[key] = dict(dtype="float64", shape=[len(names)], names=names, fps=info["fps"])
            if raw in RAW_HAND_POINTS:
                if len(names) != 63:
                    raise ValueError(f"Expected 21 xyz hand points: {raw}")
                features[key].update(coordinate_frame="source_left_camera",
                                     point_order="source order; anatomical order unconfirmed")
            stats[key] = dict(min=array.min(0).tolist(), max=array.max(0).tolist(),
                              mean=array.mean(0).tolist(), std=array.std(0).tolist(), count=[count])
            mapping[raw] = key
        index = dict(common, frame_index=pa.array(range(count)), index=pa.array(range(count)),
                     task_index=pa.array(task_ids))
        pq.write_table(pa.table(index), data / "master_index.parquet")
        converted = layout.create_empty_dataset_info(
            info["fps"], {**features, **layout.DEFAULT_FEATURES},
            master_feature=CAMERAS[next(iter(CAMERAS))], robot_type=info.get("robot_type"))
        converted.update(total_episodes=1, total_frames=count, total_tasks=len(tasks),
                         total_videos=2, splits={"train": "0:1"})
        write_json(stage / "meta/info.json", converted)
        episode = dict(episode_index=0, tasks=tasks, length=count)
        (stage / "meta/episodes.jsonl").write_text(json.dumps(episode) + "\n", encoding="utf-8")
        (stage / "meta/tasks.jsonl").write_text("".join(
            json.dumps(dict(task_index=i, task=text), ensure_ascii=False) + "\n"
            for i, text in enumerate(tasks)), encoding="utf-8")
        (stage / "meta/episodes_stats.jsonl").write_text(
            json.dumps(dict(episode_index=0, stats=stats), allow_nan=False) + "\n", encoding="utf-8")
        write_json(stage / "annotation/raw_episode_000000_structured.json",
                   structured_annotation(source, output, segments, timestamps, info["fps"]))
        manifest = dict(
            dataset_name=source.name, sequence_name="episode_000000", reference_camera="left",
            videos={}, intrinsics={}, extrinsics={"baseline_m": stereo["baseline_m"]},
            camera_features={}, stereo_pair=["head_left_rectified", "head_right_rectified"],
            frame_policy="source_order", synchronization_checked=False,
            source_episode=str(output / "data/chunk-000/episode_000000"))
        for side, key, projection in zip(("left", "right"), CAMERAS.values(), ("P1", "P2")):
            matrix = stereo[projection]
            manifest["intrinsics"][side] = dict(fx=float(matrix[0, 0]), fy=float(matrix[1, 1]),
                cx=float(matrix[0, 2]), cy=float(matrix[1, 2]))
            manifest["camera_features"][side] = key
            manifest["videos"][side] = dict(path=str(output / layout.DEFAULT_VIDEO_PATH.format(
                episode_chunk=0, episode_index=0, video_key=key)), fps=info["fps"],
                width=2160, height=2160, frame_count=count)
        write_json(stage / "calibrations/dataset_manifest.json", manifest)
        # JSON is also valid YAML, consumed by the existing safe_load readers.
        write_json(stage / "calibrations/rectify_params.yaml",
                   {key: stereo[key].tolist() for key in ("R1", "R2", "P1", "P2", "Q")})
        write_json(stage / "calibrations/source_geometry.json", dict(
            source_video="provider-confirmed undistorted pinhole",
            transform="X_right = R @ X_left + T", source_translation_unit="millimeters",
            projection_p95_px=stereo["source_projection_p95_px"],
            raw_hand_points_frame="source left camera; use R1 to rotate into rectified left camera"))
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage)
        raise
    return dict(output=str(output), frames=count, tasks=len(tasks), features=len(features))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    if (args.source / "meta/info.json").is_file():
        print(json.dumps(convert(args.source, args.output_dir, args.ffmpeg), ensure_ascii=False))
        return
    instances = sorted(args.source.glob("instance-*"))
    if not instances:
        parser.error("No instance directories found")
    failed = 0
    for source in instances:
        try:
            print(json.dumps(convert(source, args.output_dir / source.name, args.ffmpeg), ensure_ascii=False), flush=True)
        except Exception as error:
            failed += 1
            print(f"Failed {source}: {error}", file=sys.stderr, flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
