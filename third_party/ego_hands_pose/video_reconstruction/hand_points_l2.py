"""Export depth-anchored hand keypoints in HumanCOIN LeRobot format."""
from contextlib import closing
import fcntl
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

JOINTS = ["wrist", "thumbCMC", "thumbMCP", "thumbIP", "thumbTip"] + [
    finger + suffix
    for finger in ("index", "middle", "ring", "pinky")
    for suffix in ("MCP", "PIP", "DIP", "Tip")
]
COLUMNS = [f"{joint}_{axis}" for joint in JOINTS for axis in "xyz"]
CAMERA = "observation.images.headband_left_rgb_rectified"
DEPTH = "observation.images.depth_headband_left_rectified"


def export_hand_points(prediction, episode, matrix, *, destination=None, depth_offset=0):
    session = episode.parents[2]
    chunk = episode.parent.name
    depth_dir = session / "annotation" / chunk / DEPTH / episode.name
    if depth_offset:
        tag = f"m{-depth_offset}" if depth_offset < 0 else f"p{depth_offset}"
        depth_dir = depth_dir.with_name(f"{episode.name}_right_offset_{tag}")
    depth_meta = json.loads((depth_dir / "lerobot_depth.json").read_text())
    if depth_meta.get("right_frame_offset", 0) != depth_offset:
        raise ValueError(f"Unexpected depth offset: {depth_dir}")
    if depth_meta["camera_feature"] != CAMERA:
        raise ValueError(f"Depth camera mismatch: {depth_dir}")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "roboego-hand-vis"))
    try:
        from roboego_hand_vis.lerobot_fast_foundation_depth.base.depth_video import read_episode_depth
    finally:
        sys.path.pop(0)

    with np.load(prediction, allow_pickle=False) as data:
        frames = data["frame_indices"]
        joints = data["joints_3d"]
        present = data["existence_prob"] >= 0.5
    if not len(frames) or np.any(frames < 0) or np.any(np.diff(frames) <= 0):
        raise ValueError(f"Invalid prediction frame indices: {prediction}")
    source = pq.read_table(episode / f"{CAMERA}.parquet", columns=["timestamp"])
    timestamps = source["timestamp"].take(pa.array(frames)).cast(pa.float64())
    by_frame = {int(frame): i for i, frame in enumerate(frames)}
    processed = depth_meta["processed_frames"]
    available = set(processed)
    expected = available.intersection(by_frame)
    if len(available) != len(processed) or not expected:
        raise ValueError(f"Invalid or nonoverlapping depth frame indices: {depth_dir}")
    if len(expected) != len(frames):
        print(f"{episode}: {len(frames) - len(expected)} frames without depth, exporting nulls", flush=True)
    corrected = np.full_like(joints, np.nan)
    matched = set()
    matrix = np.asarray(matrix)
    with closing(read_episode_depth(depth_dir)) as depths:
        for frame, depth in zip(processed, depths):
            if frame not in by_frame:
                continue
            i = by_frame[frame]
            matched.add(frame)
            for hand in np.flatnonzero(present[i]):
                points = joints[i, hand]
                wrist = points[0]
                if not np.isfinite(points).all() or wrist[2] <= 0:
                    continue
                uv = matrix @ wrist
                x, y = np.rint(uv[:2] / uv[2]).astype(int)
                height, width = depth.shape
                if not (0 <= x < width and 0 <= y < height):
                    continue
                samples = depth[max(0, y - 5):y + 6, max(0, x - 5):x + 6]
                samples = samples[np.isfinite(samples) & (samples >= .15) & (samples <= 2)]
                if len(samples) < 25 or np.diff(np.percentile(samples, [10, 90]))[0] > .06:
                    continue
                z = np.median(samples)
                if abs(z - wrist[2]) <= .25:
                    corrected[i, hand] = points + wrist * (z / wrist[2] - 1)
            if matched == expected:
                break
    if matched != expected:
        raise ValueError(f"Depth video ended before all required frames: {depth_dir}")

    target = destination or episode
    target.mkdir(parents=True, exist_ok=True)
    index = int(episode.name.removeprefix("episode_"))
    features = {}
    with tempfile.TemporaryDirectory(dir=target) as temporary:
        staging = Path(temporary)
        for hand, side in enumerate(("left", "right")):
            feature = f"observation.hand_{side}_points"
            values = corrected[:, hand].reshape(-1, 63)
            columns = {"timestamp": timestamps, "episode_index": pa.array([index] * len(frames), type=pa.int64())}
            columns.update({
                name: pa.array(values[:, i], mask=~np.isfinite(values[:, i]), type=pa.float32())
                for i, name in enumerate(COLUMNS)
            })
            pq.write_table(pa.table(columns), staging / f"{feature}.parquet")
            features[feature] = {"dtype": "float32", "shape": [63], "names": COLUMNS}
        if destination is None:
            info_path = session / "meta/info.json"
            with (info_path.parent / ".hand_points.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                info = json.loads(info_path.read_text())
                camera = info["features"][CAMERA]
                for feature in features.values():
                    for key in ("fps", "tolerance_s"):
                        if key in camera:
                            feature[key] = camera[key]
                info["features"].update(features)
                pending = info_path.with_name(".info.hand_points.pending")
                pending.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                for file in staging.glob("*.parquet"):
                    file.replace(target / file.name)
                pending.replace(info_path)
        else:
            for file in staging.glob("*.parquet"):
                file.replace(target / file.name)
    return np.isfinite(corrected[:, :, 0, 2]).sum(axis=0).tolist()
