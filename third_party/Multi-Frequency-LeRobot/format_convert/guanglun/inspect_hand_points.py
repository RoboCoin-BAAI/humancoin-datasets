"""Visualize source-order hand points and test world-to-camera projection.

Canonical finger chains are hypotheses for inspection, not verified joint names.
No dataset features are changed.
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from format_convert.field_naming import HAND_LANDMARK_NAMES

CHAINS = [[0, *range(i, i+4)] for i in (1, 5, 9, 13, 17)]
COLORS = ["#e74c3c", "#27ae60", "#2980b9", "#9b59b6", "#e67e22"]


def rotation(q):
    v = np.array([q.get(k, 0) for k in ("x", "y", "z", "w")], dtype=float)
    if not np.isfinite(v).all() or np.linalg.norm(v) < 1e-12:
        raise ValueError("Invalid quaternion")
    x, y, z, w = v / np.linalg.norm(v)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def project(points, extrinsic, intrinsic):
    transform, = extrinsic["data"]["transforms"]
    if transform["parent_frame_id"] != "world":
        raise ValueError("Expected camera pose in world")
    r = rotation(transform["rotation"])
    t = np.array([transform["translation"].get(k, 0) for k in ("x", "y", "z")])
    camera = (np.asarray(points)-t) @ r
    k = np.array(intrinsic["data"]["K"]).reshape(3, 3)
    homogeneous = camera @ k.T
    with np.errstate(divide="ignore", invalid="ignore"):
        pixels = homogeneous[:, :2] / homogeneous[:, 2:3]
    return pixels, camera[:, 2]


def nearest(times, stamp, tolerance_ns=1_000_000):
    i = int(np.argmin(np.abs(np.asarray(times, dtype=np.int64)-stamp)))
    if abs(int(times[i])-stamp) > tolerance_ns:
        raise ValueError("Stream alignment exceeds 1 ms")
    return i


def bgr(color):
    return tuple(int(color[i:i+2], 16) for i in (5, 3, 1))


def draw_hand(image, pixels, depth, side):
    h, w = image.shape[:2]
    valid = (depth > 0) & np.isfinite(pixels).all(1)
    inside = valid & (pixels[:, 0] >= 0) & (pixels[:, 0] < w) & (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
    for chain, color in zip(CHAINS, COLORS):
        for a, b in zip(chain, chain[1:]):
            if inside[a] and inside[b]:
                cv2.line(image, tuple(np.rint(pixels[a]).astype(int)),
                         tuple(np.rint(pixels[b]).astype(int)), bgr(color), 3, cv2.LINE_AA)
    for i in range(21):
        if not inside[i]:
            continue
        x, y = np.rint(pixels[i]).astype(int)
        color = (255, 255, 255) if i == 0 else bgr(COLORS[(i-1)//4])
        cv2.circle(image, (x, y), 5, color, -1, cv2.LINE_AA)
        text = side[0].upper()+str(i)
        cv2.putText(image, text, (x+6, y-6), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, text, (x+6, y-6), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA)
    return dict(positive_depth=int(valid.sum()), inside_image=int(inside.sum()),
                camera_depth_min=float(depth.min()), camera_depth_max=float(depth.max()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--times", nargs="+", type=float, default=[5, 15, 25])
    args = parser.parse_args(argv)
    session = args.session.resolve()
    out = session/"diagnostics/hand_points"
    out.mkdir(parents=True, exist_ok=True)
    data = session/"data/chunk-000/episode_000000"
    source = json.loads((session/"meta/guanglun_source.json").read_text(encoding="utf-8"))
    origin = source["timestamp_origin_ns"]
    calibration = json.loads((session/"calibrations/source_camera_calibration.json").read_text(encoding="utf-8"))
    # Recover the native timeline from float64 relative Parquet timestamps.
    hands = {}
    for side in ("left", "right"):
        canonical = data/f"observation.hand_{side}_points.parquet"
        table = pq.read_table(canonical if canonical.exists() else data/f"observation.hand_{side}_pose.parquet")
        names = HAND_LANDMARK_NAMES if canonical.exists() else [f"transform_{i:03d}" for i in range(21)]
        xyz = np.stack([np.column_stack([table[f"{name}_{axis}"].to_numpy()
                                        for axis in ("x", "y", "z")]) for name in names], axis=1)
        times = origin + np.rint(table["timestamp"].to_numpy()*1e9).astype(np.int64)
        hands[side] = (xyz, times)
    master = pq.read_table(data/"observation.images.head_left_rectified.parquet")["timestamp"].to_numpy()
    samples, metrics = [], []
    for time in args.times:
        frame_index = int(np.argmin(abs(master-time)))
        stamp = origin+int(round(float(master[frame_index])*1e9))
        hand_points = {side: xyz[nearest(times, stamp)] for side, (xyz, times) in hands.items()}
        images = []
        record = dict(frame_index=frame_index, time_s=float(master[frame_index]), cameras={})
        for camera in ("head_left", "head_right"):
            cap = cv2.VideoCapture(str(session/f"videos/chunk-000/observation.images.{camera}_rectified/episode_000000.mp4"))
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, image = cap.read()
                if not ok:
                    raise ValueError("Video frame unavailable")
            finally:
                cap.release()
            streams = calibration[camera+"_rectified"]
            if isinstance(streams["intrinsic"], list):
                extrinsic = streams["extrinsic"][nearest([r["timestamp_ns"] for r in streams["extrinsic"]], stamp)]
                intrinsic = streams["intrinsic"][nearest([r["timestamp_ns"] for r in streams["intrinsic"]], stamp)]
            else:
                table = pq.read_table(data/(streams["world_pose_feature"]+".parquet"))
                pose_times = origin+np.rint(table["timestamp"].to_numpy()*1e9).astype(np.int64)
                pose_idx = nearest(pose_times, stamp)
                extrinsic = {"data":{"transforms":[dict(parent_frame_id="world",
                    translation={k:table[k][pose_idx].as_py() for k in ("x","y","z")},
                    rotation={k:table["q"+k][pose_idx].as_py() for k in ("x","y","z","w")})]}}
                intrinsic = {"data":streams["intrinsic"]}
            record["cameras"][camera] = {}
            for side, points in hand_points.items():
                pixels, depth = project(points, extrinsic, intrinsic)
                record["cameras"][camera][side] = draw_hand(image, pixels, depth, side)
            cv2.putText(image, f'{camera} | frame {frame_index} | source indices; candidate chains',
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
            images.append(image)
        cv2.imwrite(str(out/f"projection_{frame_index:06d}.jpg"), np.hstack(images))
        samples.append((float(master[frame_index]), hand_points))
        metrics.append(record)
    # Export static 3D plots and a fully offline rotatable HTML comparison.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    fig = plt.figure(figsize=(12, 5*len(samples)))
    interactive = make_subplots(rows=1, cols=2, specs=[[{"type":"scene"}, {"type":"scene"}]],
                                subplot_titles=["Left hand (wrist centered)", "Right hand (wrist centered)"])
    trace_groups = []
    for sample_idx, (time, points_by_side) in enumerate(samples):
        trace_ids = []
        for side_idx, side in enumerate(("left", "right")):
            points = points_by_side[side]-points_by_side[side][0]
            ax = fig.add_subplot(len(samples), 2, sample_idx*2+side_idx+1, projection="3d")
            ax.scatter(*points.T, c="black", s=12)
            for i, point in enumerate(points):
                ax.text(*point, str(i), fontsize=8)
            for chain, color in zip(CHAINS, COLORS):
                ax.plot(*points[chain].T, color=color)
                trace_ids.append(len(interactive.data))
                interactive.add_trace(go.Scatter3d(x=points[chain,0], y=points[chain,1], z=points[chain,2],
                    mode="lines", line=dict(color=color,width=5),showlegend=False,visible=sample_idx==0),row=1,col=side_idx+1)
            trace_ids.append(len(interactive.data))
            interactive.add_trace(go.Scatter3d(x=points[:,0],y=points[:,1],z=points[:,2],mode="markers+text",
                text=[str(i) for i in range(21)],textposition="top center",marker=dict(size=3,color="black"),
                customdata=HAND_LANDMARK_NAMES,hovertemplate="index=%{text}<br>candidate=%{customdata}<extra></extra>",
                showlegend=False,visible=sample_idx==0),row=1,col=side_idx+1)
            ax.set_title(f"{side} | {time:.2f}s | candidate canonical chains")
            span=max(float(np.ptp(points,axis=0).max()),.01)/2
            center=(points.max(0)+points.min(0))/2
            for fn, c in zip((ax.set_xlim,ax.set_ylim,ax.set_zlim),center):fn(c-span,c+span)
            ax.set_box_aspect((1,1,1));ax.set_xlabel("source X");ax.set_ylabel("source Y");ax.set_zlabel("source Z")
        trace_groups.append(trace_ids)
    fig.tight_layout();fig.savefig(out/"indexed_hands_3d.png", dpi=150);plt.close(fig)
    interactive.update_layout(title="Source hand indices 0-20; anatomical names/chains are hypotheses",
        scene=dict(aspectmode="data"),scene2=dict(aspectmode="data"),
        updatemenus=[dict(buttons=[dict(label=f"{time:.2f}s",method="update",
            args=[{"visible":[i in trace_groups[j] for i in range(len(interactive.data))]}])
            for j,(time,_) in enumerate(samples)])])
    interactive.write_html(str(out/"indexed_hands_3d.html"),include_plotlyjs=True)
    report=dict(method="Assume source XYZ in world; project using inverse(T_world_camera) and K; no distortion applied",
                joint_order_status="Canonical finger chains are hypotheses pending visual/provider confirmation",
                note="Positive depth and inside-image counts alone do not establish anatomical correctness.",samples=metrics)
    (out/"projection_report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))
    print(out)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
