import copy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from format_convert.guanglun.pose_qc import check, Limits, PoseJumpError
from format_convert.guanglun import convert_guanglun_mcap as converter


def records():
    times=[0,33_333_333,66_666_666]
    streams={"observation.head_pose":[(t,[0,0,0,0,0,0,1]) for t in times]}
    for side in ("left","right"):
        streams[f"observation.hand_{side}_points"]=[(t,[0.]*63) for t in times]
        streams[f"observation.hand_{side}_quaternions"]=[(t,[0,0,0,1]*21) for t in times]
    return dict(streams=streams)


def test_still_pose_and_quaternion_sign_change_pass():
    value=records()
    value["streams"]["observation.head_pose"][1][1][-1]=-1
    assert check(value)["passed"]


def test_head_translation_and_single_fingertip_jump_block():
    value=records()
    value["streams"]["observation.head_pose"][1][1][0]=.5
    report=check(value)
    assert not report["passed"]
    assert report["streams"]["observation.head_pose.position"]["examples"][0]["frame"]==1
    value=records()
    value["streams"]["observation.hand_right_points"][1][1][60]=.8
    report=check(value)
    assert not report["passed"]
    assert report["streams"]["observation.hand_right_points"]["examples"][0]["joint_index"]==20


def test_rotation_jump_and_zero_quaternion_block():
    value=records()
    value["streams"]["observation.head_pose"][1][1][3:]=[1,0,0,0]
    assert not check(value)["passed"]
    value=records()
    value["streams"]["observation.hand_left_quaternions"][1][1][:4]=[0,0,0,0]
    assert not check(value)["passed"]


def test_large_time_gap_does_not_misclassify_slow_motion():
    value=records()
    value["streams"]["observation.head_pose"]=[(0,[0,0,0,0,0,0,1]),
                                               (1_000_000_000,[.5,0,0,0,0,0,1])]
    assert check(value)["passed"]
    with pytest.raises(ValueError,match="positive"):
        check(value,Limits(head_jump_m=-1))


def test_failed_qc_blocks_before_any_output_or_video_work(tmp_path,monkeypatch):
    value=records()
    value["streams"]["observation.hand_left_points"][1][1][0]=1
    source=tmp_path/"input.mcap";source.touch()
    out=tmp_path/"new_parent/output"
    monkeypatch.setattr(converter,"read_mcap",lambda path:value)
    monkeypatch.setattr(converter,"remux",lambda *a:pytest.fail("Must not start video conversion"))
    with pytest.raises(PoseJumpError):
        converter.convert(source,out,"ffmpeg")
    assert not out.parent.exists()
