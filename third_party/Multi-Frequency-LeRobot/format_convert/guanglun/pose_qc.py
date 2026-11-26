"""Conservative adjacent-frame jump gate for source head and hand poses only."""
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Limits:
    head_jump_m: float = .40
    hand_jump_m: float = .60
    head_jump_deg: float = 120.
    hand_jump_deg: float = 160.


class PoseJumpError(ValueError):
    def __init__(self, report):
        self.report = report
        super().__init__("Severe head/hand pose jump detected; conversion blocked")


def _scan(samples, *, kind, step_limit, angle_limit=None):
    stamps = np.asarray([x[0] for x in samples], dtype=np.int64)
    values = np.asarray([x[1] for x in samples], dtype=np.float64)
    if len(samples) < 2:
        raise ValueError("At least two pose samples are required")
    dt = np.diff(stamps)/1e9
    if np.any(dt <= 0) or not np.isfinite(values).all():
        raise ValueError("Invalid pose values or timestamps")
    if kind == "quaternion":
        q = values.reshape(len(samples), -1, 4)
        norm = np.linalg.norm(q, axis=2, keepdims=True)
        if np.any(norm < 1e-8):
            raise ValueError("Zero-length pose quaternion")
        q = q/norm
        # q and -q represent the same rotation; do not flag sign changes.
        dot = np.clip(abs(np.sum(q[1:]*q[:-1],axis=2)),0,1)
        distance = np.degrees(2*np.arccos(dot))
        limit = angle_limit
    else:
        xyz = values.reshape(len(samples), -1, 3)
        distance = np.linalg.norm(xyz[1:]-xyz[:-1],axis=2)
        limit = step_limit
    # At <=30 FPS use a generous per-frame floor. Across longer source gaps,
    # also require the equivalent high velocity rather than flag ordinary motion.
    threshold = limit*np.maximum(1.,dt*30.)[:,None]
    rows, joints = np.where(distance > threshold)
    details = [dict(previous_frame=int(i),frame=int(i+1),joint_index=int(j),
                    timestamp_ns=int(stamps[i+1]),dt_s=float(dt[i]),
                    jump=float(distance[i,j]),allowed_jump=float(threshold[i,0]),
                    rate_per_s=float(distance[i,j]/dt[i])) for i,j in zip(rows[:20],joints[:20])]
    return dict(samples=len(samples),max_jump=float(distance.max()),
                max_rate_per_s=float((distance/dt[:,None]).max()),
                violations=len(rows),examples=details)


def check(records, limits=None):
    limits = limits or Limits()
    if any(not np.isfinite(v) or v <= 0 for v in asdict(limits).values()):
        raise ValueError("All pose jump limits must be finite and positive")
    streams = records["streams"]
    report = dict(passed=True,limits=asdict(limits),
                  policy="Adjacent jumps; thresholds scale up for intervals longer than 1/30 s; no repairs or dropped frames",
                  streams={})
    checks = []
    head = streams.get("observation.head_pose")
    if head:
        if any(len(row[1]) != 7 for row in head):
            report["streams"]["observation.head_pose"] = dict(error="Expected one xyz+xyzw head transform")
        else:
            checks.extend([
                ("observation.head_pose.position", [(t,v[:3]) for t,v in head], "position", limits.head_jump_m, None),
                ("observation.head_pose.rotation", [(t,v[3:]) for t,v in head], "quaternion", None, limits.head_jump_deg)])
    else:
        report["streams"]["observation.head_pose"] = dict(error="Required head pose missing")
    for side in ("left", "right"):
        for suffix, kind, width in (("points","position",63),("quaternions","quaternion",84)):
            key = f"observation.hand_{side}_{suffix}"
            samples = streams.get(key)
            if not samples or any(len(v) != width for _,v in samples):
                report["streams"][key] = dict(error=f"Required {width}-dimensional hand stream missing/invalid")
                continue
            checks.append((key,samples,kind,limits.hand_jump_m,limits.hand_jump_deg))
    for key,samples,kind,step,angle in checks:
        try:
            report["streams"][key] = _scan(samples,kind=kind,step_limit=step,angle_limit=angle)
        except ValueError as exc:
            report["streams"][key] = dict(error=str(exc))
    report["passed"] = not any(x.get("error") or x.get("violations",0) for x in report["streams"].values())
    return report
