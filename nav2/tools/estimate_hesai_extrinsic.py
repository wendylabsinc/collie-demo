#!/usr/bin/env python3
"""Estimate and qualify the Hesai-to-base rigid transform from cloud pairs.

The estimator is intentionally separate from deployment. A passing report is
evidence for review, never an automatic authorization to enable robot motion.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class Estimate:
    x_m: float
    y_m: float
    z_m: float
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    median_residual_m: float
    p90_residual_m: float
    symmetric_inlier_ratio: float
    correspondence_count: int
    floor_angle_deg: float | None
    floor_height_delta_m: float | None
    body_nonfloor_coverage: float | None
    body_nonfloor_median_residual_m: float | None
    body_nonfloor_p90_residual_m: float | None
    score: float


def rotation_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def rpy_from_rotation(rotation: np.ndarray) -> tuple[float, float, float]:
    pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
    if abs(math.cos(pitch)) > 1e-7:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = 0.0
        yaw = math.atan2(-rotation[0, 1], rotation[1, 1])
    return roll, pitch, yaw


def transform_points(
    points: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    return points @ rotation.T + translation


def voxel_downsample(points: np.ndarray, voxel_m: float) -> np.ndarray:
    if not 0 < voxel_m:
        raise ValueError("voxel size must be positive")
    keys = np.floor(points / voxel_m).astype(np.int32)
    _, first = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(first)]


def clean_cloud(
    points: np.ndarray,
    *,
    maximum_range_m: float,
    voxel_m: float,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    finite = np.isfinite(points).all(axis=1)
    radius = np.linalg.norm(points, axis=1)
    points = points[
        finite
        & (radius >= 0.30)
        & (radius <= maximum_range_m)
        & (np.abs(points[:, 2]) <= maximum_range_m)
    ]
    return voxel_downsample(points, voxel_m)


def rigid_fit(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def icp(
    source: np.ndarray,
    target: np.ndarray,
    initial_rotation: np.ndarray,
    initial_translation: np.ndarray,
    *,
    iterations: int = 45,
    initial_correspondence_m: float = 0.45,
    final_correspondence_m: float = 0.16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rotation = initial_rotation.copy()
    translation = initial_translation.copy()
    tree = cKDTree(target)
    previous_error = math.inf
    residuals = np.empty(0, dtype=np.float64)
    for step in range(iterations):
        transformed = transform_points(source, rotation, translation)
        distance, index = tree.query(transformed, k=1, workers=-1)
        fraction = step / max(1, iterations - 1)
        threshold = (
            initial_correspondence_m * (1.0 - fraction)
            + final_correspondence_m * fraction
        )
        robust_cut = min(threshold, float(np.quantile(distance, 0.72)))
        keep = distance <= max(final_correspondence_m, robust_cut)
        if int(keep.sum()) < 80:
            break
        delta_rotation, delta_translation = rigid_fit(
            transformed[keep],
            target[index[keep]],
        )
        rotation = delta_rotation @ rotation
        translation = delta_rotation @ translation + delta_translation
        residuals = distance[keep]
        error = float(np.median(residuals))
        if abs(previous_error - error) < 1e-5:
            break
        previous_error = error
    return rotation, translation, residuals


def symmetric_residuals(
    source_transformed: np.ndarray,
    target: np.ndarray,
    threshold_m: float,
) -> tuple[np.ndarray, float]:
    source_distance, _ = cKDTree(target).query(
        source_transformed,
        k=1,
        workers=-1,
    )
    target_distance, _ = cKDTree(source_transformed).query(
        target,
        k=1,
        workers=-1,
    )
    combined = np.concatenate([source_distance, target_distance])
    inliers = combined[combined <= threshold_m]
    ratio = float(inliers.size / max(1, combined.size))
    return inliers, ratio


def fit_floor(points: np.ndarray) -> tuple[np.ndarray, float] | None:
    if points.shape[0] < 100:
        return None
    cutoff = float(np.quantile(points[:, 2], 0.35))
    candidates = points[points[:, 2] <= cutoff]
    if candidates.shape[0] < 100:
        return None
    rng = np.random.default_rng(20260730)
    best = np.empty(0, dtype=np.int64)
    for _ in range(500):
        sample = candidates[rng.choice(candidates.shape[0], 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        length = float(np.linalg.norm(normal))
        if length < 1e-7:
            continue
        normal /= length
        if abs(float(normal[2])) < 0.75:
            continue
        distance = np.abs(candidates @ normal - sample[0] @ normal)
        inliers = np.flatnonzero(distance <= 0.035)
        if inliers.size > best.size:
            best = inliers
    if best.size < 80:
        return None
    floor_points = candidates[best]
    center = floor_points.mean(axis=0)
    _, _, vt = np.linalg.svd(floor_points - center, full_matrices=False)
    normal = vt[-1]
    if normal[2] < 0:
        normal = -normal
    normal /= np.linalg.norm(normal)
    offset = -float(normal @ center)
    return normal, offset


def floor_metrics(
    source_transformed: np.ndarray,
    target: np.ndarray,
) -> tuple[float | None, float | None]:
    source_plane = fit_floor(source_transformed)
    target_plane = fit_floor(target)
    if source_plane is None or target_plane is None:
        return None, None
    source_normal, source_offset = source_plane
    target_normal, target_offset = target_plane
    angle = math.degrees(
        math.acos(float(np.clip(source_normal @ target_normal, -1.0, 1.0)))
    )
    source_height = -source_offset / source_normal[2]
    target_height = -target_offset / target_normal[2]
    return angle, abs(float(source_height - target_height))


def rotation_between_vectors(
    source: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    cross = np.cross(source, target)
    cosine = float(np.clip(source @ target, -1.0, 1.0))
    sine = float(np.linalg.norm(cross))
    if sine < 1e-9:
        return np.eye(3)
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / sine**2)


def enforce_floor_alignment(
    source: np.ndarray,
    target: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Refine roll, pitch, and height from the two observed floor planes.

    Unconstrained point-to-point ICP can trade several centimetres of floor
    height for a slightly better wall fit. A mobile robot cannot afford that
    ambiguity because it would move obstacles vertically through the costmap.
    """

    source_plane = fit_floor(transform_points(source, rotation, translation))
    target_plane = fit_floor(target)
    if source_plane is None or target_plane is None:
        return rotation, translation
    source_normal, _ = source_plane
    target_normal, target_offset = target_plane
    correction = rotation_between_vectors(source_normal, target_normal)
    rotation = correction @ rotation
    translation = correction @ translation

    aligned_plane = fit_floor(transform_points(source, rotation, translation))
    if aligned_plane is None:
        return rotation, translation
    _, aligned_offset = aligned_plane
    # Translating every source point by alpha*n changes the plane offset from
    # d to d-alpha. Choose alpha=d_source-d_target.
    translation = (
        translation
        + (aligned_offset - target_offset) * target_normal
    )
    return rotation, translation


def body_nonfloor_metrics(
    source_transformed: np.ndarray,
    target: np.ndarray,
    *,
    threshold_m: float,
) -> tuple[float | None, float | None, float | None]:
    """Measure how well Hesai explains body-LiDAR vertical geometry.

    The body-frame cloud has shorter range and less coverage than the XT16, so
    the meaningful direction is body cloud to Hesai. This check excludes the
    floor; passing it requires the external sensor to reproduce walls and
    obstacle returns seen by the onboard sensor.
    """

    target_floor = fit_floor(target)
    if target_floor is None:
        return None, None, None
    normal, offset = target_floor
    source_height = source_transformed @ normal + offset
    target_height = target @ normal + offset
    source_nonfloor = source_transformed[source_height >= 0.15]
    target_nonfloor = target[target_height >= 0.15]
    if min(source_nonfloor.shape[0], target_nonfloor.shape[0]) < 100:
        return None, None, None
    distance, _ = cKDTree(source_nonfloor).query(
        target_nonfloor,
        k=1,
        workers=-1,
    )
    inliers = distance[distance <= threshold_m]
    coverage = float(inliers.size / target_nonfloor.shape[0])
    if inliers.size == 0:
        return coverage, math.inf, math.inf
    return (
        coverage,
        float(np.median(inliers)),
        float(np.quantile(inliers, 0.90)),
    )


def score_transform(
    source: np.ndarray,
    target: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    *,
    inlier_threshold_m: float,
) -> Estimate:
    transformed = transform_points(source, rotation, translation)
    residuals, inlier_ratio = symmetric_residuals(
        transformed,
        target,
        inlier_threshold_m,
    )
    if residuals.size == 0:
        median = p90 = math.inf
    else:
        median = float(np.median(residuals))
        p90 = float(np.quantile(residuals, 0.90))
    floor_angle, floor_height_delta = floor_metrics(transformed, target)
    (
        body_nonfloor_coverage,
        body_nonfloor_median,
        body_nonfloor_p90,
    ) = body_nonfloor_metrics(
        transformed,
        target,
        threshold_m=inlier_threshold_m,
    )
    roll, pitch, yaw = rpy_from_rotation(rotation)
    # Low residual alone can reward a tiny accidental overlap. Penalize low
    # symmetric coverage so the result must explain both sensors' geometry.
    score = median + 0.25 * (1.0 - inlier_ratio)
    return Estimate(
        x_m=float(translation[0]),
        y_m=float(translation[1]),
        z_m=float(translation[2]),
        roll_deg=math.degrees(roll),
        pitch_deg=math.degrees(pitch),
        yaw_deg=math.degrees(yaw),
        median_residual_m=median,
        p90_residual_m=p90,
        symmetric_inlier_ratio=inlier_ratio,
        correspondence_count=int(residuals.size),
        floor_angle_deg=floor_angle,
        floor_height_delta_m=floor_height_delta,
        body_nonfloor_coverage=body_nonfloor_coverage,
        body_nonfloor_median_residual_m=body_nonfloor_median,
        body_nonfloor_p90_residual_m=body_nonfloor_p90,
        score=score,
    )


def estimate_transform(
    source: np.ndarray,
    target: np.ndarray,
    *,
    voxel_m: float = 0.10,
    maximum_range_m: float = 7.0,
    inlier_threshold_m: float = 0.18,
    initial_xyz: Iterable[float] = (0.0, 0.0, 0.0),
    yaw_search_degrees: Iterable[float] = tuple(range(-180, 180, 30)),
) -> Estimate:
    source = clean_cloud(
        source,
        maximum_range_m=maximum_range_m,
        voxel_m=voxel_m,
    )
    target = clean_cloud(
        target,
        maximum_range_m=maximum_range_m,
        voxel_m=voxel_m,
    )
    if min(source.shape[0], target.shape[0]) < 250:
        raise ValueError("not enough overlapping geometry after filtering")

    initial_xyz = np.asarray(tuple(initial_xyz), dtype=np.float64)
    # Align the observed floor heights before the coarse yaw search. The
    # caller's z remains an additive mounting prior.
    floor_z = float(np.quantile(target[:, 2], 0.08)) - float(
        np.quantile(source[:, 2], 0.08)
    )
    candidates: list[
        tuple[Estimate, np.ndarray, np.ndarray]
    ] = []
    for yaw_degrees in yaw_search_degrees:
        initial_rotation = rotation_from_rpy(
            0.0,
            0.0,
            math.radians(float(yaw_degrees)),
        )
        for dx in (-0.20, 0.0, 0.20):
            for dy in (-0.20, 0.0, 0.20):
                translation = initial_xyz + np.array([dx, dy, floor_z])
                rotation, refined_translation, _ = icp(
                    source,
                    target,
                    initial_rotation,
                    translation,
                )
                rotation, refined_translation = enforce_floor_alignment(
                    source,
                    target,
                    rotation,
                    refined_translation,
                )
                estimate = score_transform(
                    source,
                    target,
                    rotation,
                    refined_translation,
                    inlier_threshold_m=inlier_threshold_m,
                )
                # A physically credible Go2 mount is local to the body. This
                # rejects room-alignment aliases before choosing the best fit.
                if np.linalg.norm(refined_translation) <= 1.0:
                    candidates.append(
                        (estimate, rotation, refined_translation)
                    )
    if not candidates:
        raise ValueError("ICP found no transform inside the one-metre mount bound")

    best, _, _ = min(candidates, key=lambda item: item[0].score)
    return best


def load_capture(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as capture:
        metadata = json.loads(str(capture["metadata"]))
        return (
            np.array(capture["hesai_points"], dtype=np.float64),
            np.array(capture["base_points"], dtype=np.float64),
            metadata,
        )


def angle_delta_degrees(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b + 180.0) % 360.0 - 180.0


def consistency_report(estimates: list[Estimate]) -> dict:
    translations = np.array(
        [[item.x_m, item.y_m, item.z_m] for item in estimates]
    )
    rotations = np.array(
        [
            [item.roll_deg, item.pitch_deg, item.yaw_deg]
            for item in estimates
        ]
    )
    rotation_center = np.array(
        [
            math.degrees(
                math.atan2(
                    np.sin(np.radians(rotations[:, axis])).mean(),
                    np.cos(np.radians(rotations[:, axis])).mean(),
                )
            )
            for axis in range(3)
        ]
    )
    rotation_error = angle_delta_degrees(rotations, rotation_center)
    translation_std = translations.std(axis=0)
    rotation_std = rotation_error.std(axis=0)
    metrics_pass = all(
        item.median_residual_m <= 0.10
        and item.p90_residual_m <= 0.17
        and item.symmetric_inlier_ratio >= 0.35
        and item.floor_angle_deg is not None
        and item.floor_angle_deg <= 2.0
        and item.floor_height_delta_m is not None
        and item.floor_height_delta_m <= 0.035
        and item.body_nonfloor_coverage is not None
        and item.body_nonfloor_coverage >= 0.85
        and item.body_nonfloor_median_residual_m is not None
        and item.body_nonfloor_median_residual_m <= 0.10
        and item.body_nonfloor_p90_residual_m is not None
        and item.body_nonfloor_p90_residual_m <= 0.16
        for item in estimates
    )
    consistency_pass = (
        len(estimates) >= 3
        and float(np.max(translation_std)) <= 0.035
        and float(np.max(rotation_std)) <= 2.0
    )
    return {
        "sample_count": len(estimates),
        "translation_mean_m": translations.mean(axis=0).tolist(),
        "translation_std_m": translation_std.tolist(),
        "rotation_mean_deg": rotation_center.tolist(),
        "rotation_std_deg": rotation_std.tolist(),
        "per_capture_metrics_pass": metrics_pass,
        "cross_capture_consistency_pass": consistency_pass,
        "calibration_pass": metrics_pass and consistency_pass,
        "motion_unlock_authorized": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("captures", type=Path, nargs="+")
    parser.add_argument("--voxel", type=float, default=0.10)
    parser.add_argument("--maximum-range", type=float, default=7.0)
    parser.add_argument("--inlier-threshold", type=float, default=0.18)
    parser.add_argument(
        "--initial-xyz",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, 0.0),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = []
    estimates = []
    for path in args.captures:
        source, target, metadata = load_capture(path)
        estimate = estimate_transform(
            source,
            target,
            voxel_m=args.voxel,
            maximum_range_m=args.maximum_range,
            inlier_threshold_m=args.inlier_threshold,
            initial_xyz=args.initial_xyz,
        )
        estimates.append(estimate)
        results.append(
            {
                "capture": str(path),
                "metadata": metadata,
                "estimate": asdict(estimate),
            }
        )
    report = {
        "schema": 1,
        "transform_direction": "hesai_lidar_to_base_link",
        "captures": results,
        "consistency": consistency_report(estimates),
        "note": (
            "A passing report still requires a human-reviewed floor, wall, "
            "and known-obstacle check before deployment is unlocked."
        ),
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(encoded + "\n")
    print(encoded)
    return 0 if report["consistency"]["calibration_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
