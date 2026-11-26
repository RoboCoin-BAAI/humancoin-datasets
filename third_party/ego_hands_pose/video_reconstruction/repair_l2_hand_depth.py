"""Re-export L2 hand points from timestamp-aligned stereo depth and saved predictions."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys

from hand_points_l2 import export_hand_points


def repair(item):
    session = Path(item["session"])
    episodes = sorted((session / "data").glob("chunk-*/episode_*"))
    try:
        if [episode.name for episode in episodes] != [row["episode"] for row in item["episodes"]]:
            raise ValueError("Episode list has changed since timestamp audit")
        for episode, row in zip(episodes, item["episodes"]):
            prediction = session / "annotation" / episode.parent.name / "hand_pose_left_rectified" / episode.name
            metadata = json.loads((prediction / "metadata.json").read_text())
            if metadata.get("geometry_camera_source") != "calibration":
                raise ValueError(f"Prediction does not use fixed calibration: {prediction}")
            counts = export_hand_points(
                prediction / "prediction.npz", episode, metadata["geometry_intrinsics_source"],
                depth_offset=row["offset"],
            )
            print(f"{episode}: offset={row['offset']}, valid left/right rows={counts}", flush=True)
        return 0
    except Exception as error:
        print(f"{session}: {error}", file=sys.stderr, flush=True)
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    os.environ["PATH"] = os.pathsep.join(
        filter(None, (str(Path(sys.executable).parent), os.environ.get("PATH")))
    )
    selected = [
        item for item in json.loads(args.audit.read_text())
        if item["status"] == "ok" and any(row["offset"] for row in item["episodes"])
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        failed = sum(pool.map(repair, selected))
    print(f"{len(selected)} sessions, {failed} failed")
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
