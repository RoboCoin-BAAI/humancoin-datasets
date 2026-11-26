"""Regression checks for Archebase timestamp and source-value preservation."""
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from format_convert.archebase.convert_archebase_mcap import (
    increasing, pose_vector, prepare_annotation, write_feature,
)


def test_annotation_uses_integer_nanoseconds_and_keeps_outside_camera_range(tmp_path):
    origin = 1789619246881482625
    segment = dict(startTime=origin-127, endTime=origin+1000000127,
                   description="move book", hand="both", verbs=["move"],
                   objects=["book"], referenceObjects=["tray"], actionResult="success")
    annotation = dict(annoDetail=dict(timeSegment=dict(segments=[segment])))
    result = prepare_annotation(annotation, origin, tmp_path)
    row = result["segments"][0]
    assert row["start_s"] == -127/1e9
    assert row["end_s"] == 1.000000127
    assert row["source_segment"] == segment
    assert result["source_annotation"] == annotation


def test_native_feature_timestamps_and_bool_int_types(tmp_path):
    origin = 1789619246881482625
    samples = [(origin-127, [False]), (origin+1, [True])]
    write_feature(tmp_path, "valid", samples, origin, dict(dtype="bool",names=["valid"]))
    table = pq.read_table(tmp_path/"data/chunk-000/episode_000000/valid.parquet")
    assert table["timestamp"].to_pylist() == [-127/1e9, 1/1e9]
    assert table["valid"].to_pylist() == [False, True]
    write_feature(tmp_path, "track", [(origin, [-1])], origin, dict(dtype="int64",names=["id"]))
    table = pq.read_table(tmp_path/"data/chunk-000/episode_000000/track.parquet")
    assert table["id"].to_pylist() == [-1]


def test_pose_preserves_quaternion_and_source_axes():
    from types import SimpleNamespace as NS
    pose = NS(position=NS(x=1,y=-2,z=3),orientation=NS(x=.1,y=.2,z=.3,w=.4))
    np.testing.assert_array_equal(pose_vector(pose), [1,-2,3,.1,.2,.3,.4])


@pytest.mark.parametrize("times", [[], [1,1], [2,1]])
def test_invalid_native_timeline_is_rejected(times):
    with pytest.raises(ValueError,match="timestamps"):
        increasing(times,"sample")
