"""Compact native stereo calibration without inventing rectification matrices."""
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml


def rotation(q):
    v = np.array([q.get(k, 0) for k in ("x", "y", "z", "w")], dtype=float)
    norm = np.linalg.norm(v)
    if not np.isfinite(v).all() or norm < 1e-12:
        raise ValueError("Invalid camera quaternion")
    x, y, z, w = v / norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def pose(row):
    value, = row["data"]["transforms"]
    if value["parent_frame_id"] != "world":
        raise ValueError("Expected camera poses in world")
    transform = np.eye(4)
    transform[:3,:3] = rotation(value["rotation"])
    transform[:3,3] = [value["translation"].get(k, 0) for k in ("x", "y", "z")]
    return transform


def compact(source):
    output = dict(schema_version="1.0", calibration_type="native_rectified_stereo",
                  rectification_performed_by_converter=False,
                  camera_pose_note="Per-frame world poses are trajectories, not changing stereo calibration.")
    for camera, pose_feature in (("head_left", "observation.head_camera_pose"),
                                 ("head_right", "observation.right_eye_camera_pose")):
        key = camera+"_rectified"
        rows = source[key]
        intrinsic = None
        if rows["intrinsic"]:
            intrinsic = {k:v for k,v in rows["intrinsic"][0]["data"].items() if k != "timestamp"}
            for row in rows["intrinsic"]:
                actual = {k:v for k,v in row["data"].items() if k != "timestamp"}
                if actual != intrinsic:
                    raise ValueError(f"Intrinsic calibration varies: {camera}")
        output[key] = dict(raw_camera_id=camera, feature_key="observation.images."+key,
                           intrinsic=intrinsic, world_pose_feature=pose_feature,
                           source_intrinsic_samples=len(rows["intrinsic"]),
                           source_extrinsic_samples=len(rows["extrinsic"]))
    left, right = source["head_left_rectified"]["extrinsic"], source["head_right_rectified"]["extrinsic"]
    if not left or len(left) != len(right):
        output["stereo"] = None
        return output
    transforms = []
    for l, r in zip(left, right):
        if l["timestamp_ns"] != r["timestamp_ns"]:
            raise ValueError("Camera pose timestamps differ")
        transforms.append(np.linalg.inv(pose(l)) @ pose(r))
    transforms = np.asarray(transforms)
    first = transforms[0]
    translation_change = float(np.linalg.norm(transforms[:,:3,3]-first[:3,3],axis=1).max())
    rotation_change = float(np.max(abs(transforms[:,:3,:3]-first[:3,:3])))
    if translation_change > 1e-6 or rotation_change > 1e-5:
        raise ValueError("Stereo relative transform is not constant")
    output["stereo"] = dict(
        T_left_right=first.tolist(), T_right_left=np.linalg.inv(first).tolist(),
        transform_convention="T_left_right maps right-camera coordinates into left-camera coordinates",
        baseline_source_units=float(np.linalg.norm(first[:3,3])),
        baseline_m=float(np.linalg.norm(first[:3,3])), translation_unit="m",
        metric_unit_note="Source translation unit confirmed as metres by user.",
        relative_rotation_deg=float(np.degrees(np.arccos(np.clip((np.trace(first[:3,:3])-1)/2,-1,1)))),
        validated_frame_pairs=len(left), max_translation_change_source_units=translation_change,
        max_rotation_matrix_change=rotation_change)
    return output


def export(root, final_root, value, videos):
    directory = root/"calibrations"
    directory.mkdir(parents=True, exist_ok=True)
    (directory/"source_camera_calibration.json").write_text(json.dumps(value,indent=2)+"\n",encoding="utf-8")
    if value["stereo"] is None or any(value[c+"_rectified"]["intrinsic"] is None for c in ("head_left","head_right")):
        return
    intrinsics, manifest_videos, cameras, time_axes = {}, {}, {}, []
    for side in ("left", "right"):
        key = "head_"+side+"_rectified"
        feature = "observation.images."+key
        intrinsic = value[key]["intrinsic"]
        k = intrinsic["K"]
        intrinsics[side] = dict(fx=k[0],fy=k[4],cx=k[2],cy=k[5])
        video = videos[feature]
        manifest_videos[side] = dict(path=str(final_root/f"videos/chunk-000/{feature}/episode_000000.mp4"),
                                     fps=video["fps"],frame_count=video["decoded_frames"],
                                     width=video["width"],height=video["height"])
        cameras[key] = dict(camera_model="pinhole",feature_key=feature,
                            intrinsics=[k[0],k[4],k[2],k[5]],resolution=[video["width"],video["height"]],
                            distortion_model=intrinsic.get("distortion_model","not_provided"),
                            distortion_coeffs=intrinsic.get("D",[]))
        time_axes.append(pq.read_table(root/f"data/chunk-000/episode_000000/{feature}.parquet")["timestamp"].to_pylist())
    if len(time_axes[0]) != len(time_axes[1]):
        raise ValueError("Stereo frame counts differ")
    manifest = dict(dataset_name=final_root.name,sequence_name="episode_000000",reference_camera="left",
        videos=manifest_videos,intrinsics=intrinsics,extrinsics=value["stereo"],
        source_episode=str(final_root/"data/chunk-000/episode_000000"),stereo_pair=["head_left","head_right"],
        frame_policy="source_order",synchronization_checked=True,
        max_stereo_timestamp_difference_s=max(abs(a-b) for a,b in zip(*time_axes)),
        camera_features={side:"observation.images.head_"+side+"_rectified" for side in ("left","right")},
        rectification_origin="native_source",metric_depth_ready=True,
        metric_depth_note="Metric baseline available. Native cameras retain slight intrinsic/rotation differences; depth conversion uses the native rectified pair approximation.")
    (directory/"dataset_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    params = dict(cameras=cameras,frame_policy="source_order",native_rectified=True,
                  rectification_performed_by_converter=False,
                  relative_extrinsics=value["stereo"],
                  note="Native calibration only. No new R1/R2/P1/P2/Q matrices were computed or applied.")
    (directory/"rectify_params.yaml").write_text(yaml.safe_dump(params,sort_keys=False),encoding="utf-8")
