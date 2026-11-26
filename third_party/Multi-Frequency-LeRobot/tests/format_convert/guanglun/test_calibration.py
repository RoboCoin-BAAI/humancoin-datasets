from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from format_convert.guanglun.calibration import compact


def sample():
    value={}
    for side, offset in (("left",0),("right",.064)):
        key="head_"+side+"_rectified"
        value[key]=dict(intrinsic=[],extrinsic=[])
        for i in range(3):
            value[key]["intrinsic"].append(dict(timestamp_ns=i,data=dict(timestamp=str(i),K=[100,0,50,0,100,40,0,0,1])))
            value[key]["extrinsic"].append(dict(timestamp_ns=i,data=dict(transforms=[dict(
                parent_frame_id="world",child_frame_id=key,translation=dict(x=i+offset),rotation=dict(w=1))])))
    return value


def test_moving_world_pose_becomes_fixed_stereo_extrinsics():
    result=compact(sample())
    assert result["head_left_rectified"]["source_extrinsic_samples"]==3
    assert isinstance(result["head_left_rectified"]["intrinsic"],dict)
    assert result["stereo"]["baseline_source_units"]==pytest.approx(.064)
    assert result["stereo"]["baseline_m"] == pytest.approx(.064)
    assert result["stereo"]["translation_unit"] == "m"
    np.testing.assert_allclose(np.array(result["stereo"]["T_left_right"])[:3,3],[.064,0,0])


def test_varying_intrinsics_and_nonrigid_rig_not_collapsed():
    value=sample();value["head_left_rectified"]["intrinsic"][1]["data"]["K"][0]=101
    with pytest.raises(ValueError,match="Intrinsic"):
        compact(value)
    value=sample();value["head_right_rectified"]["extrinsic"][1]["data"]["transforms"][0]["translation"]["x"]+=.01
    with pytest.raises(ValueError,match="not constant"):
        compact(value)
