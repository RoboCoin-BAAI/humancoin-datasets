import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from format_convert.guanglun.format_semantics import format_segments, main


def test_mapping_does_not_invent_missing_semantics(tmp_path):
    source = tmp_path/"segments.json"
    source.write_text(json.dumps([
        dict(task_description="Clean", segment=dict(subtask_description="Wipe", end_time=2)),
        dict(task_description="Clean", segment=dict(subtask_description="Put cloth down",
                                                   start_time=2, end_time=3, skill=["place"])),
    ]))
    data = format_segments(source, tmp_path)
    assert data["subtask"][0]["start_s"] == 0
    assert data["subtask"][0]["arm"] is None
    assert data["subtask"][0]["object"] is None
    assert data["subtask"][0]["progress"] is None
    assert data["finish_reason"] is None
    assert data["scene_classification"]["category"] is None
    assert data["actions"]["vocabulary"] == ["put", "wipe"]
    assert data["subtask"][0]["labels"] == ["wipe"]
    assert data["subtask"][1]["labels"] == ["put"]
    assert data["task_description"] == "Clean"
    assert data["timesteps_s"] == [0, 2, 3]
    assert data["source"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert "annotation_provenance" not in data
    assert "capture_context" not in data


def test_invalid_intervals_rejected(tmp_path):
    source = tmp_path/"segments.json"
    source.write_text('[{"segment":{"start_time":2,"end_time":1,"subtask_description":"x"}}]')
    with pytest.raises(ValueError, match="interval"):
        format_segments(source, tmp_path)


def test_existing_annotation_not_overwritten(tmp_path):
    (tmp_path/"annotation").mkdir()
    output = tmp_path/"annotation/episode_000000_structured.json"
    output.write_text("keep")
    with pytest.raises(SystemExit):
        main(["--session", str(tmp_path)])
    assert output.read_text() == "keep"
