import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from .camera import validate_intrinsics

REQUIRED_FIELDS = (
    "global_orient",
    "hand_pose",
    "betas",
    "translation",
    "joints_3d",
    "anchors_2d",
    "existence_prob",
    "visibility_prob",
    "predicted_intrinsics",
    "frame_indices",
    "timestamps",
)


@dataclass(frozen=True)
class ConsistencyOptions:
    existence_threshold: float = 0.5
    visibility_threshold: float = 0.2
    huber_pixels: float = 8.0
    translation_prior_xy_m: float = 0.03
    translation_prior_z_m: float = 0.1
    max_relative_wrist_depth_change: float = 0.25
    min_depth_m: float = 0.02
    iterations: int = 12
    chunk_size: int = 128

    def validate(self):
        if not all((np.isfinite(v) for v in asdict(self).values())):
            raise ValueError("Nonfinite global consistency option")
        if not all((0 <= v <= 1 for v in (self.existence_threshold, self.visibility_threshold))):
            raise ValueError("Confidence thresholds must be in [0,1]")
        if (
            min(
                self.huber_pixels,
                self.translation_prior_xy_m,
                self.translation_prior_z_m,
                self.min_depth_m,
            )
            <= 0
            or not 0 < self.max_relative_wrist_depth_change < 1
            or self.iterations < 1
            or (self.chunk_size < 1)
        ):
            raise ValueError("Invalid global consistency solver option")


def _weighted_median(values, weights):
    order = np.argsort(values, axis=0)
    sorted_values = np.take_along_axis(values, order, axis=0)
    cumulative = np.cumsum(weights[order], axis=0)
    index = (cumulative >= weights.sum() * 0.5).argmax(axis=0)
    return sorted_values[index, np.arange(values.shape[1])]


def robust_location(values, weights, scale_floor):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values).all(-1) & np.isfinite(weights) & (weights > 0)
    values, weights = (values[valid], weights[valid])
    if not len(values):
        raise ValueError("No finite positive-weight samples")
    center = _weighted_median(values, weights)
    scale = np.maximum(1.4826 * _weighted_median(abs(values - center), weights), scale_floor)
    for _ in range(20):
        distance = np.sqrt(np.mean(((values - center) / scale) ** 2, axis=-1))
        effective = weights * np.minimum(1, 1.5 / np.maximum(distance, 1e-12))
        updated = (values * effective[:, None]).sum(0) / effective.sum()
        if np.max(abs(updated - center)) < 1e-09:
            center = updated
            break
        center = updated
    return center


def source_transforms(metadata, count):
    matrix = metadata.get("source_to_canvas")
    if matrix is None:
        if "crop_affine" not in metadata:
            raise ValueError("Missing source-to-canvas transform")
        left, top, sx, sy = metadata["crop_affine"]
        matrix = [[sx, 0, -left * sx], [0, sy, -top * sy], [0, 0, 1]]
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape == (3, 3):
        matrix = np.broadcast_to(matrix, (count, 3, 3)).copy()
    if matrix.shape != (count, 3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Invalid source-to-canvas transform")
    if (
        np.any(matrix[:, [0, 1], [0, 1]] <= 0)
        or not np.allclose(matrix[:, 2], [0, 0, 1])
        or (not np.allclose(matrix[:, [0, 1], [1, 0]], 0))
    ):
        raise ValueError("Only positive axis-aligned crop/scale/pad transforms are supported")
    return matrix


def _project_numpy(points, camera):
    focal = camera[:, [0, 1], [0, 1]][:, None, None]
    center = camera[:, :2, 2][:, None, None]
    return points[..., :2] / np.maximum(points[..., 2:], 1e-08) * focal + center


def _stats(values):
    values = np.asarray(values)
    if not values.size:
        return dict(count=0, median=None, p95=None, maximum=None)
    return dict(
        count=int(values.size),
        median=float(np.median(values)),
        p95=float(np.quantile(values, 0.95)),
        maximum=float(np.max(values)),
    )


def refine_translation(
    canonical,
    old_joints,
    original,
    anchors_px,
    camera,
    old_camera,
    reliability,
    options,
    geometry_min_z=None,
):
    dtype, device = (canonical.dtype, canonical.device)
    focal = camera[..., [0, 1], [0, 1]][:, None, :].expand(-1, 2, -1).reshape(-1, 2)
    center = camera[..., :2, 2][:, None, :].expand(-1, 2, -1).reshape(-1, 2)
    old_f = old_camera[..., [0, 1], [0, 1]][:, None, :].expand(-1, 2, -1).reshape(-1, 2)
    old_c = old_camera[..., :2, 2][:, None, :].expand(-1, 2, -1).reshape(-1, 2)
    points = canonical.reshape(-1, 21, 3)
    targets = anchors_px.reshape(-1, 21, 2)
    prior = original.reshape(-1, 3)
    wrist = old_joints.reshape(-1, 21, 3)[:, 0]
    old_uv = wrist[:, :2] / wrist[:, 2:].clamp_min(options.min_depth_m) * old_f + old_c
    transported = (
        torch.cat(((old_uv - center) / focal * wrist[:, 2:], wrist[:, 2:]), -1) - points[:, 0]
    )
    minimum_z = points[..., 2].amin(-1) if geometry_min_z is None else geometry_min_z.reshape(-1)
    lower = torch.maximum(
        wrist[:, 2] * (1 - options.max_relative_wrist_depth_change) - points[:, 0, 2],
        options.min_depth_m - minimum_z,
    )
    upper = torch.maximum(
        wrist[:, 2] * (1 + options.max_relative_wrist_depth_change) - points[:, 0, 2], lower + 1e-05
    )
    transported[:, 2] = transported[:, 2].clamp(lower, upper)
    transported[:, :2] = (old_uv - center) / focal * (
        points[:, 0, 2:] + transported[:, 2:]
    ) - points[:, 0, :2]
    confidence = reliability.reshape(-1)
    relative = (targets - center[:, None]) / focal[:, None]
    finite = torch.isfinite(targets).all(-1) & torch.isfinite(relative).all(-1)
    targets = torch.nan_to_num(targets)
    votes = finite.to(dtype) * confidence[:, None]
    extent = torch.linalg.vector_norm(
        torch.where(finite[..., None], targets, -1000000000.0).amax(-2)
        - torch.where(finite[..., None], targets, 1000000000.0).amin(-2),
        dim=-1,
    )
    eligible = (finite.sum(-1) >= 6) & (confidence > 0) & (extent >= 8)
    inv_sigma2 = torch.tensor(
        [options.translation_prior_xy_m ** (-2)] * 2 + [options.translation_prior_z_m ** (-2)],
        device=device,
        dtype=dtype,
    )

    def evaluate(translation):
        placed = points + translation[:, None]
        uv = (
            placed[..., :2] / placed[..., 2:].clamp_min(options.min_depth_m) * focal[:, None]
            + center[:, None]
        )
        residual = uv - targets
        norm = torch.linalg.vector_norm(residual, dim=-1)
        delta = options.huber_pixels
        rho = torch.where(norm <= delta, 0.5 * norm.square(), delta * (norm - 0.5 * delta))
        cost = (rho * votes).sum(-1) + 0.5 * ((translation - prior).square() * inv_sigma2).sum(-1)
        return (cost, residual, placed)

    translation = transported.clone()
    alternative = prior.clone()
    alternative[:, 2] = alternative[:, 2].clamp(lower, upper)
    choose_prior = eligible & (evaluate(alternative)[0] < evaluate(translation)[0])
    translation = torch.where(choose_prior[:, None], alternative, translation)
    for _ in range(options.iterations):
        cost, residual, placed = evaluate(translation)
        z = placed[..., 2].clamp_min(options.min_depth_m)
        derivative = torch.zeros(*residual.shape, 3, device=device, dtype=dtype)
        derivative[..., 0, 0] = focal[:, None, 0] / z
        derivative[..., 1, 1] = focal[:, None, 1] / z
        derivative[..., :, 2] = -placed[..., :2] / z[..., None].square() * focal[:, None]
        robust = (
            options.huber_pixels / torch.linalg.vector_norm(residual, dim=-1).clamp_min(1e-08)
        ).clamp_max(1)
        weight = votes * robust
        flat_j = derivative.reshape(len(points), -1, 3)
        flat_w = weight[..., None].expand_as(residual).reshape(len(points), -1)
        hessian = torch.bmm(flat_j.transpose(1, 2), flat_j * flat_w[..., None])
        hessian += torch.diag(inv_sigma2)[None]
        gradient = (derivative * (weight[..., None] * residual)[..., None]).sum((1, 2))
        gradient += (translation - prior) * inv_sigma2
        step = torch.linalg.solve(hessian, -gradient[..., None])[..., 0]
        accepted = ~eligible
        for fraction in (1.0, 0.5, 0.25, 0.125, 0.0625):
            candidate = translation + fraction * step
            candidate[:, 2] = candidate[:, 2].clamp(lower, upper)
            new_cost = evaluate(candidate)[0]
            use = ~accepted & torch.isfinite(candidate).all(-1) & (new_cost < cost)
            translation = torch.where(use[:, None], candidate, translation)
            accepted |= use
    translation = torch.where(eligible[:, None], translation, transported)
    residual = evaluate(translation)[1]
    rms = ((residual.square().sum(-1) * votes).sum(-1) / votes.sum(-1).clamp_min(1e-08)).sqrt()
    return (
        translation.reshape_as(original),
        eligible.reshape_as(reliability),
        rms.reshape_as(reliability),
    )


@torch.inference_mode()
def apply_global_consistency(
    arrays, metadata, mano, *, device="cpu", options=None, fixed_intrinsics_source=None
):
    started = time.perf_counter()
    options = options or ConsistencyOptions()
    options.validate()
    fixed_intrinsics_source = validate_intrinsics(fixed_intrinsics_source)
    if metadata.get("architecture") != "parametric_pinhole":
        raise ValueError("Global consistency requires rectified parametric pinhole predictions")
    if metadata.get("overlay_camera") == "supplied":
        raise ValueError("Supplied visualization K cannot be used as predicted geometry K")
    if "global_consistency_segment_ids" in arrays:
        raise ValueError("Prediction has already been globally postprocessed")
    for name in REQUIRED_FIELDS:
        if name not in arrays or not np.isfinite(arrays[name]).all():
            raise ValueError(f"Missing or nonfinite field: {name}")
    count = len(arrays["frame_indices"])
    expected = dict(
        global_orient=(count, 2, 3),
        hand_pose=(count, 2, 45),
        betas=(count, 2, 10),
        translation=(count, 2, 3),
        joints_3d=(count, 2, 21, 3),
        anchors_2d=(count, 2, 21, 2),
        existence_prob=(count, 2),
        visibility_prob=(count, 2),
        predicted_intrinsics=(count, 3, 3),
        frame_indices=(count,),
        timestamps=(count,),
    )
    if not count or any((np.shape(arrays[k]) != shape for k, shape in expected.items())):
        raise ValueError("Unexpected full-sequence prediction shape")
    starts = [0]
    if (
        not starts
        or starts[0] != 0
        or any((not isinstance(s, (int, np.integer)) for s in starts))
        or (starts != sorted(set(starts)))
        or (starts[-1] >= count)
    ):
        raise ValueError("Segment starts must be sorted unique row offsets beginning at zero")
    camera = np.asarray(arrays["predicted_intrinsics"], dtype=np.float64)
    if (
        np.any(camera[:, [0, 1], [0, 1]] <= 0)
        or not np.allclose(camera[:, 2], [0, 0, 1])
        or (not np.allclose(camera[:, [0, 1], [1, 0]], 0))
    ):
        raise ValueError("Invalid rectified pinhole camera")
    for name in ("predicted_camera_valid", "overlay_camera_valid"):
        if name in arrays and (not np.asarray(arrays[name]).all()):
            raise ValueError("Cannot postprocess a sequence containing invalid cameras")
    transforms = source_transforms(metadata, count)
    inverse = np.linalg.inv(transforms)
    source_k = inverse @ camera
    source_size = np.asarray(metadata.get("source_size", metadata["size"]), dtype=np.float64)
    size = np.asarray(metadata["size"], dtype=np.float64)
    if (
        source_size.shape != (2,)
        or size.shape != (2,)
        or (not np.isfinite(source_size).all())
        or (not np.isfinite(size).all())
        or (min(*source_size, *size) <= 0)
    ):
        raise ValueError("Invalid image dimensions")
    e, v = (np.asarray(arrays[k], dtype=np.float64) for k in ("existence_prob", "visibility_prob"))
    if ((e < 0) | (e > 1) | (v < 0) | (v > 1)).any():
        raise ValueError("Confidence is outside [0,1]")
    hand_weight = (
        e**2 * v**2 * (e >= options.existence_threshold) * (v >= options.visibility_threshold)
    )
    old_uv = _project_numpy(np.asarray(arrays["joints_3d"]), camera)
    anchor = np.asarray(arrays["anchors_2d"])
    target = anchor * size
    votes = ((anchor > 0.02) & (anchor < 0.98)).all(-1) & (
        arrays["joints_3d"][..., 2] > options.min_depth_m
    )
    residual2 = np.sum((old_uv - target) ** 2, axis=-1)
    raw_rms = np.sqrt(np.sum(np.where(votes, residual2, 0), axis=-1) / np.maximum(votes.sum(-1), 1))
    high = np.where(votes[..., None], target, -1000000000.0).max(-2)
    low = np.where(votes[..., None], target, 1000000000.0).min(-2)
    threshold = np.maximum(15, 0.25 * np.linalg.norm(high - low, axis=-1))
    wrist_inside = ((old_uv[..., 0, :] >= 0) & (old_uv[..., 0, :] < size)).all(-1)
    geometry_reliable = (votes.sum(-1) >= 6) & (raw_rms <= threshold) & wrist_inside
    rejected_geometry = (hand_weight > 0) & ~geometry_reliable
    hand_weight *= geometry_reliable
    new_beta = np.empty_like(arrays["betas"])
    new_k = np.empty_like(camera)
    segment_ids = np.empty(count, dtype=np.int32)
    segments, shared_beta, shared_k = ([], [], [])
    for index, (start, stop) in enumerate(zip(starts, starts[1:] + [count])):
        weight = hand_weight[start:stop]
        beta = []
        shape_fallback = []
        for hand in range(2):
            trusted = weight[:, hand]
            shape_fallback.append(not bool(trusted.sum()))
            if not trusted.sum():
                trusted = np.ones(stop - start)
            beta.append(robust_location(arrays["betas"][start:stop, hand], trusted, 0.05))
        beta = np.asarray(beta, dtype=np.float32)
        k = source_k[start:stop]
        normalized = np.concatenate(
            (np.log(k[:, [0, 1], [0, 1]] / source_size), k[:, :2, 2] / source_size), -1
        )
        camera_weight = weight.max(-1)
        camera_fallback = not bool(camera_weight.sum())
        if camera_fallback:
            camera_weight = np.ones(stop - start)
        if fixed_intrinsics_source is None:
            params = robust_location(normalized, camera_weight, [0.01, 0.01, 0.002, 0.002])
            global_k = np.eye(3)
            global_k[[0, 1], [0, 1]] = np.exp(params[:2]) * source_size
            global_k[:2, 2] = params[2:] * source_size
        else:
            global_k = fixed_intrinsics_source.copy()
        new_beta[start:stop] = beta
        new_k[start:stop] = transforms[start:stop] @ global_k
        segment_ids[start:stop] = index
        shared_beta.append(beta)
        shared_k.append(global_k)
        segments.append(
            dict(
                start_row=start,
                stop_row=stop,
                first_frame_id=int(arrays["frame_indices"][start]),
                last_frame_id=int(arrays["frame_indices"][stop - 1]),
                trusted_shape_frames=(weight > 0).sum(0).tolist(),
                shape_fallback_unweighted=shape_fallback,
                camera_fallback_unweighted=camera_fallback,
            )
        )
    joints, vertices, translation, refined, rms = ([], [], [], [], [])
    device = torch.device(device)
    for start in range(0, count, options.chunk_size):
        stop = min(count, start + options.chunk_size)

        def tensor(value):
            return torch.as_tensor(np.array(value, copy=True), device=device, dtype=torch.float32)

        orient, pose, beta, prior = (
            tensor(value[start:stop])
            for value in (
                arrays["global_orient"],
                arrays["hand_pose"],
                new_beta,
                arrays["translation"],
            )
        )
        canonical, mesh = mano(orient[None], pose[None], beta[None], torch.zeros_like(prior)[None])
        canonical, mesh = (canonical[0], mesh[0])
        anchor = np.asarray(arrays["anchors_2d"][start:stop], dtype=np.float64)
        target = anchor * size
        target = np.where(((anchor > 0.02) & (anchor < 0.98)).all(-1)[..., None], target, np.nan)
        t, used, residual = refine_translation(
            canonical,
            tensor(arrays["joints_3d"][start:stop]),
            prior,
            tensor(target),
            tensor(new_k[start:stop]),
            tensor(camera[start:stop]),
            tensor(hand_weight[start:stop]),
            options,
            geometry_min_z=torch.minimum(canonical[..., 2].amin(-1), mesh[..., 2].amin(-1)),
        )
        joints.append((canonical + t[..., None, :]).cpu().numpy())
        vertices.append((mesh + t[..., None, :]).cpu().numpy())
        translation.append(t.cpu().numpy())
        refined.append(used.cpu().numpy())
        rms.append(residual.cpu().numpy())
    result = dict(arrays)
    result.update(
        betas=new_beta,
        translation=np.concatenate(translation),
        joints_3d=np.concatenate(joints),
        vertices=np.concatenate(vertices),
        predicted_intrinsics=new_k.astype(np.float32),
        overlay_intrinsics=new_k.astype(np.float32),
        global_consistency_segment_ids=segment_ids,
        global_betas=np.asarray(shared_beta),
        global_intrinsics_source=np.asarray(shared_k),
        global_consistency_refined=np.concatenate(refined),
        global_consistency_rms_px=np.concatenate(rms),
        raw_betas=np.asarray(arrays["betas"]),
        raw_translation=np.asarray(arrays["translation"]),
        raw_predicted_intrinsics=np.asarray(arrays["predicted_intrinsics"]),
    )
    for name in ("betas", "translation", "joints_3d", "vertices", "predicted_intrinsics"):
        if not np.isfinite(result[name]).all():
            raise ValueError(f"Global consistency generated nonfinite {name}")
    old = np.asarray(arrays["joints_3d"], dtype=np.float64)
    new = result["joints_3d"].astype(np.float64)
    old_uv, new_uv = (_project_numpy(old, camera), _project_numpy(new, new_k))
    delta_source = np.einsum("tij,thkj->thki", inverse[:, :2, :2], new_uv - old_uv)
    visible = (
        (e >= options.existence_threshold)[..., None]
        & (old[..., 2] > options.min_depth_m)
        & (new[..., 2] > options.min_depth_m)
    )
    target = np.asarray(arrays["anchors_2d"]) * size
    anchor_valid = ((arrays["anchors_2d"] > 0.02) & (arrays["anchors_2d"] < 0.98)).all(-1) & visible
    report = dict(
        method="global_shape_camera_translation",
        options=asdict(options),
        assumptions=[
            "one identity per hand slot per segment",
            "fixed rectified physical pinhole K per segment",
        ],
        camera_coordinates="shared in original pixels; mapped to canvas using recorded affine",
        network_redecoded=False,
        pose_changed=False,
        existence_visibility_anchors_changed=False,
        window_diagnostics_scope="raw network outputs before this geometric postprocess",
        fallback="preserve raw projected wrist and wrist depth under new shape/K; enforce positive geometry depth",
        segments=segments,
        refined_frame_hands=int(result["global_consistency_refined"].sum()),
        raw_geometry_rejected_frame_hands=int(rejected_geometry.sum()),
        projection_drift_source_px=_stats(np.linalg.norm(delta_source, axis=-1)[visible]),
        trusted_projection_drift_source_px=_stats(
            np.linalg.norm(delta_source, axis=-1)[visible & (hand_weight > 0)[..., None]]
        ),
        joints_drift_mm=_stats(
            np.linalg.norm(new - old, axis=-1)[
                np.broadcast_to(e[..., None] >= options.existence_threshold, old.shape[:-1])
            ]
            * 1000
        ),
        reprojection_before_canvas_px=_stats(
            np.linalg.norm(old_uv - target, axis=-1)[anchor_valid]
        ),
        reprojection_after_canvas_px=_stats(np.linalg.norm(new_uv - target, axis=-1)[anchor_valid]),
        translation_change_mm=_stats(
            np.linalg.norm(result["translation"] - arrays["translation"], axis=-1)[
                e >= options.existence_threshold
            ]
            * 1000
        ),
        newly_nonpositive_visible_joints=int(
            (
                (e[..., None] >= options.existence_threshold)
                & (old[..., 2] > options.min_depth_m)
                & (new[..., 2] <= 0)
            ).sum()
        ),
        seconds=time.perf_counter() - started,
    )
    if fixed_intrinsics_source is not None:
        report.update(
            method="global_shape_fixed_calibrated_camera_translation",
            camera_source="calibration",
            camera_aggregated=False,
        )
    return (result, report)
