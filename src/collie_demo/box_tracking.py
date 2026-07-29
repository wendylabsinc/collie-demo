from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
import math
import time

import cv2
import numpy as np
from numpy.typing import NDArray


BBoxXYWH = tuple[float, float, float, float]
TrackerUpdate = tuple[bool, BBoxXYWH]
TrackerFactory = Callable[[object, tuple[int, int, int, int]], "KLTBoxTracker"]


class BoxTrackerInitializationError(ValueError):
    """Raised when a box cannot seed a reliable image track."""


@dataclass(frozen=True, slots=True)
class KLTBoxTrackerConfig:
    """Bounded settings for the selected-fruit image tracker.

    The tracker intentionally emits no confidence score. YOLO remains the
    class and confidence authority; this tracker only fills the spatial gap
    between detector results.
    """

    max_corners: int = 96
    target_corners: int = 32
    minimum_points: int = 6
    quality_level: float = 0.01
    minimum_corner_distance_px: float = 3.0
    feature_padding_ratio: float = 0.05
    optical_flow_window_px: int = 21
    pyramid_levels: int = 3
    maximum_lk_error: float = 30.0
    maximum_forward_backward_error_px: float = 1.5
    ransac_reprojection_px: float = 3.0
    minimum_scale_per_frame: float = 0.70
    maximum_scale_per_frame: float = 1.40
    maximum_center_step_fraction: float = 0.35
    new_box_weight: float = 1.0
    minimum_box_size_px: float = 6.0

    def __post_init__(self) -> None:
        if self.max_corners < 4:
            raise ValueError("max_corners must be at least 4")
        if not self.minimum_points <= self.target_corners <= self.max_corners:
            raise ValueError(
                "target_corners must be between minimum_points and max_corners"
            )
        if self.minimum_points < 3:
            raise ValueError("minimum_points must be at least 3")
        if not 0.0 < self.quality_level <= 1.0:
            raise ValueError("quality_level must be in (0, 1]")
        if self.minimum_corner_distance_px <= 0.0:
            raise ValueError("minimum_corner_distance_px must be positive")
        if not 0.0 <= self.feature_padding_ratio <= 1.0:
            raise ValueError("feature_padding_ratio must be between 0 and 1")
        if self.optical_flow_window_px < 3:
            raise ValueError("optical_flow_window_px must be at least 3")
        if self.pyramid_levels < 0:
            raise ValueError("pyramid_levels cannot be negative")
        if self.maximum_lk_error <= 0.0:
            raise ValueError("maximum_lk_error must be positive")
        if self.maximum_forward_backward_error_px <= 0.0:
            raise ValueError(
                "maximum_forward_backward_error_px must be positive"
            )
        if self.ransac_reprojection_px <= 0.0:
            raise ValueError("ransac_reprojection_px must be positive")
        if not 0.0 < self.minimum_scale_per_frame <= 1.0:
            raise ValueError("minimum_scale_per_frame must be in (0, 1]")
        if self.maximum_scale_per_frame < 1.0:
            raise ValueError("maximum_scale_per_frame must be at least 1")
        if not 0.0 < self.maximum_center_step_fraction <= 1.0:
            raise ValueError(
                "maximum_center_step_fraction must be in (0, 1]"
            )
        if not 0.0 < self.new_box_weight <= 1.0:
            raise ValueError("new_box_weight must be in (0, 1]")
        if self.minimum_box_size_px <= 0.0:
            raise ValueError("minimum_box_size_px must be positive")


@dataclass(slots=True)
class KLTBoxTrackerMetrics:
    updates: int = 0
    accepted_updates: int = 0
    rejected_updates: int = 0
    points_in_use: int = 0
    last_update_ms: float | None = None
    total_update_ms: float = 0.0
    last_failure: str = ""

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["mean_update_ms"] = (
            None
            if self.updates == 0
            else round(self.total_update_ms / self.updates, 3)
        )
        if self.last_update_ms is not None:
            result["last_update_ms"] = round(self.last_update_ms, 3)
        result["total_update_ms"] = round(self.total_update_ms, 3)
        return result


class KLTBoxTracker:
    """Fast, confidence-neutral box tracking between YOLO detections.

    Feature motion is estimated by pyramidal Lucas-Kanade optical flow. A
    forward/backward check rejects unstable features and RANSAC estimates one
    bounded partial-affine transform for the box. The tracker fails closed when
    it lacks enough evidence or sees a physically implausible one-frame jump.
    """

    def __init__(
        self,
        bgr: object,
        bbox_xywh: tuple[int, int, int, int],
        *,
        config: KLTBoxTrackerConfig | None = None,
    ) -> None:
        self.config = config or KLTBoxTrackerConfig()
        gray = _as_gray_u8(bgr)
        self._height, self._width = gray.shape
        self._bbox = _clip_bbox(
            tuple(float(value) for value in bbox_xywh),
            self._width,
            self._height,
            self.config.minimum_box_size_px,
        )
        self._previous_gray = gray
        self._points = self._seed_points(gray, self._bbox)
        self._failed = False
        self.metrics = KLTBoxTrackerMetrics(points_in_use=len(self._points))

    def update(self, bgr: object) -> TrackerUpdate:
        started = time.perf_counter()
        self.metrics.updates += 1
        if self._failed:
            return self._reject(started, "tracker_already_failed")

        gray = _as_gray_u8(bgr)
        if gray.shape != self._previous_gray.shape:
            self._failed = True
            return self._reject(started, "frame_shape_changed")

        next_points, forward_status, forward_error = cv2.calcOpticalFlowPyrLK(
            self._previous_gray,
            gray,
            self._points,
            None,
            winSize=(
                self.config.optical_flow_window_px,
                self.config.optical_flow_window_px,
            ),
            maxLevel=self.config.pyramid_levels,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                20,
                0.01,
            ),
        )
        if (
            next_points is None
            or forward_status is None
            or forward_error is None
        ):
            self._failed = True
            return self._reject(started, "forward_flow_unavailable")

        backward_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            gray,
            self._previous_gray,
            next_points,
            None,
            winSize=(
                self.config.optical_flow_window_px,
                self.config.optical_flow_window_px,
            ),
            maxLevel=self.config.pyramid_levels,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                20,
                0.01,
            ),
        )
        if backward_points is None or backward_status is None:
            self._failed = True
            return self._reject(started, "backward_flow_unavailable")

        previous_flat = self._points.reshape(-1, 2)
        next_flat = next_points.reshape(-1, 2)
        backward_flat = backward_points.reshape(-1, 2)
        forward_backward_error = np.linalg.norm(
            previous_flat - backward_flat, axis=1
        )
        valid = (
            forward_status.reshape(-1).astype(bool)
            & backward_status.reshape(-1).astype(bool)
            & (forward_error.reshape(-1) <= self.config.maximum_lk_error)
            & (
                forward_backward_error
                <= self.config.maximum_forward_backward_error_px
            )
        )
        previous_good = previous_flat[valid]
        next_good = next_flat[valid]
        if len(next_good) < self.config.minimum_points:
            self._failed = True
            return self._reject(started, "too_few_bidirectional_points")

        transform, inlier_mask = cv2.estimateAffinePartial2D(
            previous_good,
            next_good,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.config.ransac_reprojection_px,
            maxIters=100,
            confidence=0.99,
            refineIters=5,
        )
        if transform is None or inlier_mask is None:
            self._failed = True
            return self._reject(started, "affine_estimate_failed")

        inliers = inlier_mask.reshape(-1).astype(bool)
        next_inliers = next_good[inliers]
        if len(next_inliers) < self.config.minimum_points:
            self._failed = True
            return self._reject(started, "too_few_ransac_inliers")

        scale = math.hypot(float(transform[0, 0]), float(transform[1, 0]))
        if not (
            self.config.minimum_scale_per_frame
            <= scale
            <= self.config.maximum_scale_per_frame
        ):
            self._failed = True
            return self._reject(started, "implausible_scale")

        raw_bbox = _transform_bbox(self._bbox, transform)
        try:
            clipped_bbox = _clip_bbox(
                raw_bbox,
                self._width,
                self._height,
                self.config.minimum_box_size_px,
            )
        except BoxTrackerInitializationError:
            self._failed = True
            return self._reject(started, "box_left_frame")

        old_center = _bbox_center(self._bbox)
        new_center = _bbox_center(clipped_bbox)
        center_step = math.dist(old_center, new_center)
        frame_diagonal = math.hypot(self._width, self._height)
        if (
            center_step
            > frame_diagonal * self.config.maximum_center_step_fraction
        ):
            self._failed = True
            return self._reject(started, "implausible_center_step")

        self._bbox = _blend_bbox(
            self._bbox,
            clipped_bbox,
            self.config.new_box_weight,
        )
        self._previous_gray = gray
        self._points = next_inliers.reshape(-1, 1, 2).astype(np.float32)
        if len(self._points) < self.config.target_corners:
            self._points = self._replenish_points(
                gray, self._bbox, self._points
            )

        self.metrics.accepted_updates += 1
        self.metrics.points_in_use = len(self._points)
        self.metrics.last_failure = ""
        self._finish_timing(started)
        return True, self._bbox

    def status(self) -> dict[str, object]:
        return {
            "backend": "opencv_klt_affine",
            "failed": self._failed,
            "bbox_xywh": tuple(round(value, 3) for value in self._bbox),
            "metrics": self.metrics.to_dict(),
        }

    def _seed_points(
        self, gray: NDArray[np.uint8], bbox: BBoxXYWH
    ) -> NDArray[np.float32]:
        points = _detect_features(gray, bbox, self.config)
        if points is None or len(points) < self.config.minimum_points:
            count = 0 if points is None else len(points)
            raise BoxTrackerInitializationError(
                "selected box has too little image texture for tracking "
                f"({count} points; need {self.config.minimum_points})"
            )
        return points.astype(np.float32)

    def _replenish_points(
        self,
        gray: NDArray[np.uint8],
        bbox: BBoxXYWH,
        existing: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        candidates = _detect_features(gray, bbox, self.config)
        if candidates is None:
            return existing
        accepted = [point for point in existing.reshape(-1, 2)]
        minimum_distance = self.config.minimum_corner_distance_px
        for candidate in candidates.reshape(-1, 2):
            if len(accepted) >= self.config.max_corners:
                break
            if all(
                float(np.linalg.norm(candidate - current))
                >= minimum_distance
                for current in accepted
            ):
                accepted.append(candidate)
        return np.asarray(accepted, dtype=np.float32).reshape(-1, 1, 2)

    def _reject(self, started: float, reason: str) -> TrackerUpdate:
        self.metrics.rejected_updates += 1
        self.metrics.last_failure = reason
        self.metrics.points_in_use = 0 if self._failed else len(self._points)
        self._finish_timing(started)
        return False, self._bbox

    def _finish_timing(self, started: float) -> None:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.metrics.last_update_ms = elapsed_ms
        self.metrics.total_update_ms += elapsed_ms


def produce_tracker_factory_from_mode(
    mode: str,
    *,
    config: KLTBoxTrackerConfig | None = None,
) -> TrackerFactory | None:
    """Resolve the opt-in tracker without changing the default stage path."""

    normalized = mode.casefold().strip().replace("-", "_")
    if normalized in {"", "off", "none", "disabled", "yolo"}:
        return None
    if normalized not in {"klt", "klt_affine", "opencv_klt"}:
        raise ValueError(
            "COLLIE_PRODUCE_TRACKER must be off or klt_affine"
        )

    def factory(
        bgr: object, bbox_xywh: tuple[int, int, int, int]
    ) -> KLTBoxTracker:
        return KLTBoxTracker(bgr, bbox_xywh, config=config)

    return factory


def _as_gray_u8(image: object) -> NDArray[np.uint8]:
    if not isinstance(image, np.ndarray):
        raise TypeError("tracker frame must be a numpy array")
    if image.dtype != np.uint8:
        raise TypeError("tracker frame must use uint8 pixels")
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("tracker frame must be grayscale or BGR")
    if gray.shape[0] < 2 or gray.shape[1] < 2:
        raise ValueError("tracker frame is too small")
    return np.ascontiguousarray(gray)


def _detect_features(
    gray: NDArray[np.uint8],
    bbox: BBoxXYWH,
    config: KLTBoxTrackerConfig,
) -> NDArray[np.float32] | None:
    x, y, width, height = bbox
    padding_x = width * config.feature_padding_ratio
    padding_y = height * config.feature_padding_ratio
    left = max(0, int(math.floor(x - padding_x)))
    top = max(0, int(math.floor(y - padding_y)))
    right = min(gray.shape[1], int(math.ceil(x + width + padding_x)))
    bottom = min(gray.shape[0], int(math.ceil(y + height + padding_y)))
    mask = np.zeros_like(gray)
    mask[top:bottom, left:right] = 255
    return cv2.goodFeaturesToTrack(
        gray,
        maxCorners=config.max_corners,
        qualityLevel=config.quality_level,
        minDistance=config.minimum_corner_distance_px,
        mask=mask,
        blockSize=7,
        useHarrisDetector=False,
    )


def _clip_bbox(
    bbox: BBoxXYWH,
    frame_width: int,
    frame_height: int,
    minimum_size: float,
) -> BBoxXYWH:
    x, y, width, height = bbox
    values = (x, y, width, height)
    if not all(math.isfinite(value) for value in values):
        raise BoxTrackerInitializationError("box contains a non-finite value")
    if width < minimum_size or height < minimum_size:
        raise BoxTrackerInitializationError("selected box is too small")
    left = max(0.0, min(float(frame_width), x))
    top = max(0.0, min(float(frame_height), y))
    right = max(0.0, min(float(frame_width), x + width))
    bottom = max(0.0, min(float(frame_height), y + height))
    clipped_width = right - left
    clipped_height = bottom - top
    if clipped_width < minimum_size or clipped_height < minimum_size:
        raise BoxTrackerInitializationError(
            "selected box is outside the camera frame"
        )
    return left, top, clipped_width, clipped_height


def _transform_bbox(
    bbox: BBoxXYWH, transform: NDArray[np.float64]
) -> BBoxXYWH:
    x, y, width, height = bbox
    corners = np.asarray(
        [
            [x, y],
            [x + width, y],
            [x + width, y + height],
            [x, y + height],
        ],
        dtype=np.float64,
    )
    transformed = cv2.transform(corners.reshape(1, -1, 2), transform)[0]
    left, top = np.min(transformed, axis=0)
    right, bottom = np.max(transformed, axis=0)
    return (
        float(left),
        float(top),
        float(right - left),
        float(bottom - top),
    )


def _bbox_center(bbox: BBoxXYWH) -> tuple[float, float]:
    x, y, width, height = bbox
    return x + width / 2.0, y + height / 2.0


def _blend_bbox(
    previous: BBoxXYWH, current: BBoxXYWH, current_weight: float
) -> BBoxXYWH:
    previous_weight = 1.0 - current_weight
    return tuple(
        previous_weight * old + current_weight * new
        for old, new in zip(previous, current)
    )  # type: ignore[return-value]
