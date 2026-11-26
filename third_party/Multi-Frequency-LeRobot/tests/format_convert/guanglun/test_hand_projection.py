from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from format_convert.guanglun.inspect_hand_points import project, nearest


def test_world_camera_pose_is_inverted_for_projection():
    extrinsic = {"data": {"transforms": [dict(parent_frame_id="world",
        translation=dict(x=1, y=2, z=3), rotation=dict(w=1))]}}
    intrinsic = {"data": {"K": [100, 0, 50, 0, 100, 40, 0, 0, 1]}}
    pixels, depth = project([[1, 2, 5], [2, 2, 5]], extrinsic, intrinsic)
    np.testing.assert_allclose(pixels, [[50, 40], [100, 40]])
    np.testing.assert_allclose(depth, [2, 2])


def test_alignment_preserves_nanosecond_resolution():
    origin = 1785145575556999936
    assert nearest([origin, origin+33_333_333], origin+33_333_333) == 1
    with pytest.raises(ValueError, match="1 ms"):
        nearest([origin], origin+2_000_000)
