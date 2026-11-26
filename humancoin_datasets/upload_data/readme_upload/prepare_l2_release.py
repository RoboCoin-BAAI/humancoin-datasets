"""Prepare local upload documents for merged HumanCOIN L2 datasets."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

if __package__:
    from .gen_readme import PROJECT_CONFIG, generate_portal_yaml, read_json, render_readme
    from ._common import plan_files, repo_id_for
else:
    from gen_readme import PROJECT_CONFIG, generate_portal_yaml, read_json, render_readme
    from _common import plan_files, repo_id_for

RELEASE_FILES = ("README.md", "portal.yaml", "preview.jpg")


def sensor_streams(info, desc):
    cameras = {c["id"]: c for c in desc["cameras"]["items"]}
    streams = {s["feature_key"]: s for group in desc["modalities"].values()
               for s in group.get("streams", [])}
    features = info["features"]
    result = []
    for key in dict.fromkeys([*features, *streams]):
        if key not in cameras and key not in streams:
            continue
        definition = key.removesuffix("_raw")
        if key.endswith("_emg_denoised"):
            definition = key.removesuffix("_denoised") + ".denoised"
        feature = features.get(key, features.get(definition, {}))
        stream = streams.get(key, {})
        camera = cameras.get(key, {})
        result.append({
            "feature_key": key,
            "dtype": feature.get("dtype", "float32"),
            "shape": feature.get("shape") or stream.get("shape"),
            "sample_rate_hz": stream.get("sample_rate_hz") or feature.get("fps")
                              or feature.get("sample_rate") or camera.get("fps"),
        })
    return result


def prepare(folder, project, namespace):
    desc = read_json(folder / "meta/readme.json")
    info = read_json(folder / "meta/info.json")
    if folder.name != desc["dataset_name"] or not folder.name.startswith("l2_"):
        raise ValueError(f"Unexpected L2 dataset directory: {folder}")
    primary = desc["cameras"]["primary_camera"]
    camera = next(c for c in desc["cameras"]["items"] if c["id"] == primary)
    preview = info["video_path"].format(episode_chunk=0, episode_index=0, video_key=primary)
    video = (folder / preview).resolve()
    if not video.is_relative_to(folder) or not video.is_file() or "depth" in primary:
        raise ValueError(f"Missing RGB preview video: {video}")
    payload = plan_files(folder, ignore=RELEASE_FILES)
    features = info["features"]
    groups = desc["modalities"]
    items = desc["cameras"]["items"]
    modalities = [name for name, present in (
        ("RGB", any("depth" not in c["id"] for c in items)),
        ("Depth", any("depth" in c["id"] for c in items)),
        ("Hand Pose", bool(groups.get("hand_estimation", {}).get("streams"))),
        ("Gaze", bool(groups.get("eye_gaze", {}).get("streams"))),
        ("EMG", bool(groups.get("emg", {}).get("streams"))),
        ("IMU", bool(groups.get("imu", {}).get("streams"))),
        ("Audio", any(f.get("dtype") == "audio" for f in features.values())),
    ) if present]
    portal = {
        "schema_version": 1,
        "repo_id": repo_id_for(folder, namespace),
        "dataset_name": desc["dataset_name"],
        "level": "L2",
        "license": project.get("license", "cc-by-nc-4.0"),
        "scene": desc["scene"],
        "task_name": desc["common_task_name"],
        "task_labels": [desc["task_name"]],
        "example_task_description": desc.get("task_description"),
        "example_recording": "episode_000000",
        "num_recordings": info["total_episodes"],
        "frame_count": info["total_frames"],
        "duration_s": desc["duration_s"],
        "file_count": len(payload),
        "size_bytes": sum((folder / p).stat().st_size for p in payload),
        "resolution": f'{camera["width"]}x{camera["height"]}',
        "modalities": modalities,
        "primary_camera": primary,
        "preview_video": preview,
        "sensor_streams": sensor_streams(info, desc),
    }
    readme = render_readme(folder, project, namespace, ignore=RELEASE_FILES)
    readme = readme.replace("(excluding README.md)", "(excluding README.md, portal.yaml and preview.jpg)")
    yaml_text = generate_portal_yaml(folder, portal)
    (folder / "README.md").write_text(readme, encoding="utf-8")
    (folder / "portal.yaml").write_text(yaml_text, encoding="utf-8")
    return info["total_episodes"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--namespace", default="HumanCOIN")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    folders = sorted(p.resolve() for p in args.root.glob("l2_*") if p.is_dir())
    if not folders or args.workers < 1:
        parser.error("Expected L2 datasets and a positive worker count")
    project = read_json(PROJECT_CONFIG)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        counts = list(pool.map(lambda folder: prepare(folder, project, args.namespace), folders))
    print(f"Prepared {len(counts)} datasets, {sum(counts)} episodes")


if __name__ == "__main__":
    main()
