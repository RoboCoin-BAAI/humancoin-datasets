"""Merge same-name L2 sessions into episode-indexed HumanCOIN datasets."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import errno
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def share_file(source, target):
    try:
        os.link(source, target)
    except OSError as exc:
        if exc.errno not in (errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP, errno.EMLINK):
            raise
        shutil.copy2(source, target)


def episode_path(relative, episode, *, multi_episode=True):
    parts = relative.parts
    if parts[0] == "calibrations":
        name = relative.name
        if multi_episode and not name.startswith("chunk-"):
            name = f"chunk-000_{episode}_{name}"
        return relative.with_name(name)
    return Path(*(re.sub(r"episode_000000(?=$|[_.])", episode, part) for part in parts))


def rewrite(value, paths, number, dataset, key=None):
    if isinstance(value, dict):
        return {k: rewrite(v, paths, number, dataset, k) for k, v in value.items()}
    if isinstance(value, list):
        return [rewrite(item, paths, number, dataset, key) for item in value]
    if key == "episode_index" and type(value) is int:
        return number
    if key == "task_index" and type(value) is int:
        return 0
    if key == "dataset_name":
        return dataset
    if isinstance(value, str):
        if key == "sequence_name" and value == "episode_000000":
            return f"episode_{number:06d}"
        return paths.get(value, value)
    return value


def parquet(source, target, number, offset):
    schema = pq.read_schema(source)
    names = schema.names
    if not {"episode_index", "task_index", "index"}.intersection(names):
        share_file(source, target)
        return
    table = pq.read_table(source)
    for name, value in (("episode_index", number), ("task_index", 0)):
        if name in names:
            index = names.index(name)
            field = schema.field(index)
            table = table.set_column(index, field, pa.repeat(pa.scalar(value, type=field.type), table.num_rows))
    if source.name == "master_index.parquet":
        index = names.index("index")
        field = schema.field(index)
        table = table.set_column(index, field, pa.array(range(offset, offset + table.num_rows), type=field.type))
    pq.write_table(table, target)


def merge_group(group, output_root):
    sessions = sorted(p for p in group.iterdir() if p.is_dir() and p.name.startswith("L2-"))
    if not sessions:
        raise ValueError(f"No sessions: {group}")
    destination = output_root / group.name
    if destination.exists():
        raise FileExistsError(destination)
    meta_rows = {}
    frame_offset = 0
    features = {}
    total_videos = 0
    camera_samples = {}
    modality_samples = {}
    info_template = read(sessions[0] / "meta/info.json")
    readme = read(sessions[0] / "meta/readme.json")
    task = readme["task_name"]
    with tempfile.TemporaryDirectory(prefix=f".{group.name}.", dir=output_root) as temp:
        stage = Path(temp) / group.name
        (stage / "meta").mkdir(parents=True)
        for number, source in enumerate(sessions):
            episode = f"episode_{number:06d}"
            info = read(source / "meta/info.json")
            desc = read(source / "meta/readme.json")
            if (info["total_episodes"] != 1 or desc["dataset_name"] != group.name
                    or desc["task_name"] != task or info["fps"] != info_template["fps"]
                    or info["master_feature"] != info_template["master_feature"]):
                raise ValueError(f"Incompatible session: {source}")
            for name, definition in info["features"].items():
                if name not in features:
                    features[name] = dict(definition)
                    continue
                shared = features[name]
                for k in ("dtype", "shape", "names"):
                    if k in shared and k in definition and shared[k] != definition[k]:
                        raise ValueError(f"Incompatible feature {name}.{k}: {source}")
                features[name] = {k: v for k, v in shared.items() if definition.get(k) == v}
            episode_row, = read_lines(source / "meta/episodes.jsonl")
            length = pq.ParquetFile(source / "data/chunk-000/episode_000000/master_index.parquet").metadata.num_rows
            if length != episode_row["length"] or length != info["total_frames"]:
                raise ValueError(f"Inconsistent master length: {source}")
            files = [p for folder in ("data", "videos", "audios", "calibrations", "annotation")
                     for p in (source / folder).rglob("*") if p.is_file() and not p.name.startswith(".")]
            relatives = {p.relative_to(source): episode_path(p.relative_to(source), episode, multi_episode=len(sessions) > 1)
                         for p in files}
            if len(set(relatives.values())) != len(relatives):
                raise ValueError(f"Conflicting episode files: {source}")
            paths = {}
            for old, new in relatives.items():
                paths[old.as_posix()] = new.as_posix()
                paths[str(source / old)] = str(destination / new)
            for old in ("data/chunk-000/episode_000000",):
                paths[old] = paths.get(old, episode_path(Path(old), episode).as_posix())
                paths[str(source / old)] = str(destination / paths[old])
            def relocate(value):
                if isinstance(value, dict):
                    return {k: relocate(v) for k, v in value.items()}
                if isinstance(value, list):
                    return [relocate(v) for v in value]
                if isinstance(value, str) and value.startswith("/") and str(value) not in paths:
                    marker = f"/{source.name}/"
                    if marker in value:
                        relative = value.split(marker, 1)[1]
                        if relative in paths:
                            return str(destination / paths[relative])
                return value

            for old, new in relatives.items():
                origin, target = source / old, stage / new
                target.parent.mkdir(parents=True, exist_ok=True)
                if old.suffix == ".parquet":
                    parquet(origin, target, number, frame_offset)
                elif old.suffix == ".json":
                    write(target, relocate(rewrite(read(origin), paths, number, group.name)))
                else:
                    share_file(origin, target)
                total_videos += old.parts[0] == "videos" and old.suffix == ".mp4"
            for p in sorted((source / "meta").glob("*.jsonl")):
                if p.name == "tasks.jsonl":
                    continue
                for row in read_lines(p):
                    row = relocate(rewrite(row, paths, number, group.name))
                    if p.name == "episodes.jsonl":
                        row["tasks"] = [task]
                    meta_rows.setdefault(p.name, []).append(row)
            meta_rows.setdefault("source_sessions.jsonl", []).append(dict(
                episode_index=number, session=source.name, source_tasks=episode_row["tasks"],
                source_path=str(source), pi_export=info.get("pi_export"),
            ))
            for c in desc["cameras"]["items"]:
                key = c["id"]
                camera_samples[key] = camera_samples.get(key, 0) + (c.get("sample_count") or 0)
            for group_meta in desc["modalities"].values():
                for stream in group_meta.get("streams", []):
                    key = stream["feature_key"]
                    modality_samples[key] = modality_samples.get(key, 0) + (stream.get("sample_count") or 0)
            frame_offset += length
        meta_rows["tasks.jsonl"] = [dict(task_index=0, task=task)]
        for name, rows in meta_rows.items():
            with (stage / "meta" / name).open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        info_template.pop("pi_export", None)
        info_template.update(features=features, total_episodes=len(sessions), total_frames=frame_offset,
                             total_tasks=1, total_chunks=1, total_videos=total_videos,
                             splits={"train": f"0:{len(sessions)}"})
        write(stage / "meta/info.json", info_template)
        for camera in readme["cameras"]["items"]:
            camera["sample_count"] = camera_samples[camera["id"]]
        for group_meta in readme["modalities"].values():
            for stream in group_meta.get("streams", []):
                stream["sample_count"] = modality_samples[stream["feature_key"]]
        files = [p for p in stage.rglob("*") if p.is_file()]
        readme.update(frame_count=frame_offset, duration_s=frame_offset / info_template["fps"],
                      file_count=len(files), size_bytes=sum(p.stat().st_size for p in files))
        write(stage / "meta/readme.json", readme)
        stage.rename(destination)
    return len(sessions), frame_offset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        parser.error("Source and output must be separate directories")
    groups = sorted(p for p in source.glob("l2_*") if p.is_dir() and (not args.group or p.name == args.group))
    if not groups or args.workers < 1:
        parser.error("Expected dataset groups and positive worker count")
    output.mkdir(parents=True, exist_ok=True)
    def run(group):
        try:
            count, frames = merge_group(group, output)
            print(f"{group.name}: {count} episodes, {frames} master rows", flush=True)
            return count
        except Exception as exc:
            print(f"FAILED {group.name}: {exc}", flush=True)
            return 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        totals = list(pool.map(run, groups))
    print(f"{sum(totals)} episodes in {sum(n > 0 for n in totals)} datasets", flush=True)
    return int(not all(totals))


if __name__ == "__main__":
    raise SystemExit(main())
