"""Map provider segments to the structured annotation layout without inference."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def format_segments(source: Path, episode_dir: Path, *, source_reference=None):
    raw = source.read_bytes()
    rows = json.loads(raw.decode("utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Semantic annotations must be an array")
    subtasks, boundaries, descriptions = [], set(), []
    for row in rows:
        segment = row["segment"]
        # Omitted protobuf scalars represent zero, not an unknown timestamp.
        start = float(segment.get("start_time", 0))
        end = float(segment.get("end_time", 0))
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise ValueError(f"Invalid semantic interval: {start}, {end}")
        action = segment.get("subtask_description", "")
        if not isinstance(action, str) or not action.strip():
            raise ValueError("Missing source subtask description")
        skills = segment.get("skill", [])
        if not isinstance(skills, list) or any(not isinstance(s, str) for s in skills):
            raise ValueError("Expected source skill string array")
        description = row.get("task_description")
        if description and description not in descriptions:
            descriptions.append(description)
        verb = action.split()[0].lower()
        boundaries.update((start, end))
        subtasks.append(dict(start_s=start, end_s=end, arm=None, action=action,
                             object=None, destination=None, spatial_relation=None,
                             contribution=None, progress=None, hands_visible=None,
                             hands_wearing=None, labels=[verb]))
    return dict(
        schema_version="1.0", episode_dir=str(episode_dir),
        source=dict(file=str(source_reference or source), sha256=hashlib.sha256(raw).hexdigest()),
        timesteps_s=sorted(boundaries), finish_reason=None, objects=[], scene=None,
        subtask=subtasks, task_description="\n".join(descriptions) or None,
        key_events=[], state_changes=[], tasks=[], scene_graph=[], recovery=[],
        instruction_variants=[], data_issues=[], viewpoint=None, performance_review=None,
        actions=dict(vocabulary=sorted({s for task in subtasks for s in task["labels"]})),
        scene_classification=dict(category=None, area=None, original_description=None),

    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, type=Path)
    args = parser.parse_args(argv)
    session = args.session.resolve()
    source = session / "annotation/semantic_segments.json"
    output = session / "annotation/episode_000000_structured.json"
    if output.exists():
        parser.error(f"Refusing to overwrite: {output}")
    value = format_segments(source, session / "data/chunk-000/episode_000000")
    with output.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
