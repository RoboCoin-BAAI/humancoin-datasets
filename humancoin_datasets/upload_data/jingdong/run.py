"""Upload a processed batch grouped by area and common_task_name."""

import argparse
import json
import math
from pathlib import Path
import re

from humancoin_datasets.upload_data.ks3 import KS3, atomic_json, prefix
from humancoin_datasets.upload_data.readme_upload.check_load import validate_dataset_load


def plan(batch, destination):
    batch = Path(batch).expanduser().resolve()
    destination = prefix(destination)
    sessions = sorted(path for path in batch.glob("instance-*") if path.is_dir())
    if not sessions:
        raise ValueError(f"No instance directories found: {batch}")
    items = []
    for session in sessions:
        if session.is_symlink():
            raise ValueError(f"Dataset must not be a symlink: {session}")
        annotation = session / "annotation/episode_000000_structured.json"
        labels = json.loads(annotation.read_text(encoding="utf-8"))
        area = labels["scene_classification"]["area"]
        task = labels["common_task_name"]
        for value in (area, task):
            if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", value):
                raise ValueError(f"Invalid area or common_task_name in {annotation}")
        if not task.endswith("_task"):
            raise ValueError(f"Expected a common_task_name ending in _task: {annotation}")
        if not re.fullmatch(r"instance-[A-Za-z0-9]+", session.name):
            raise ValueError(f"Invalid instance name: {session.name}")
        info = json.loads((session / "meta/info.json").read_text(encoding="utf-8"))
        camera = info["features"]["observation.images.head_left_rectified"]
        shape = camera.get("shape", [])
        fps = camera.get("fps", info.get("fps"))
        if (camera.get("dtype") != "video" or len(shape) != 3
                or any(type(value) is not int or value <= 0 for value in shape[:2])
                or type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0):
            raise ValueError(f"Invalid rectified left video specification: {session}")
        height, width = shape[:2]
        group = f"l1_{area}_{task}_{width}x{height}_{fps:g}fps"
        items.append(dict(local=str(session), destination=destination + group +
                          "/" + session.name + "/", info=info))
    return items


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", required=True, type=Path)
    parser.add_argument("--destination", required=True, help="KS3 publication root")
    parser.add_argument("--ks3-tool", type=Path, default=Path("/opt/ks3/ks3util-2.18"))
    parser.add_argument("--ks3-config", type=Path, default=Path("/opt/ks3/ks3util-config-team"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    # Build the whole batch plan before any remote writes.
    items = plan(args.batch_dir, args.destination)
    for item in items:
        print(f'{item["local"]} -> {item["destination"]}', flush=True)
    if args.dry_run:
        return 0
    client = KS3(args.ks3_tool, args.ks3_config)
    report_path = args.batch_dir.expanduser().resolve().with_name(args.batch_dir.name + "_upload.json")
    results = []
    failed = 0
    for item in items:
        local = Path(item["local"])
        result = dict(local=str(local), destination=item["destination"])
        try:
            validate_dataset_load(local, "HumanCOIN/" + local.name, item["info"])
            result.update(client.upload_directory(local, item["destination"], recursive_copy=True))
            result["status"] = "done"
        except Exception as error:
            failed += 1
            result.update(status="failed", error=str(error))
        results.append(result)
        atomic_json(report_path, dict(results=results, failed=failed, total=len(items)))
        print(json.dumps(result, ensure_ascii=False), flush=True)
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
