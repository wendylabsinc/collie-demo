from __future__ import annotations

import json
import math
import os
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response

from .shadow_server import RemoteDetection, parse_detections
from .temporal_fusion import (
    BBoxXYWH,
    MaxMojoTemporalFusion,
    TemporalFusionBackend,
    TemporalFusionRequest,
    TemporalFusionResult,
    VelocityXYWH,
)

UI_PATH = Path(__file__).with_name("max_vision_ui.html")
CLASS_COLORS = {
    "apple": (84, 112, 255),
    "banana": (56, 217, 255),
    "pear": (118, 242, 127),
}
DEFAULT_COLOR = (237, 199, 85)


@dataclass(slots=True)
class TrackState:
    label: str
    bbox_xywh: BBoxXYWH
    velocity_xywh_s: VelocityXYWH
    confidence: float
    last_verified_at: float
    last_update_at: float
    last_detector_frame_id: int | None
    last_raw_center: tuple[float, float] | None = None
    last_fused_detection_center: tuple[float, float] | None = None
    measurement_source: str = "yolo"
    alpha: float = 0.0
    center_step_px: float = 0.0


def best_detection_per_label(
    detections: tuple[RemoteDetection, ...],
) -> dict[str, RemoteDetection]:
    best: dict[str, RemoteDetection] = {}
    for detection in detections:
        current = best.get(detection.label)
        if current is None or detection.confidence > current.confidence:
            best[detection.label] = detection
    return best


def center_xy(bbox_xywh: BBoxXYWH) -> tuple[float, float]:
    x, y, width, height = bbox_xywh
    return x + width * 0.5, y + height * 0.5


def rms(values: deque[float]) -> float | None:
    if not values:
        return None
    return math.sqrt(sum(value * value for value in values) / len(values))


class MaxVisionRuntime:
    """Read-only MAX/Mojo vision fork fed by Collie's existing camera and YOLO."""

    def __init__(
        self,
        *,
        source_url: str,
        fusion: TemporalFusionBackend,
        loop_hz: float = 12.0,
        maximum_prediction_age_s: float = 0.45,
    ) -> None:
        if maximum_prediction_age_s <= 0.0:
            raise ValueError("maximum prediction age must be positive")
        self.source_url = source_url.rstrip("/")
        self.fusion = fusion
        self.loop_hz = max(1.0, float(loop_hz))
        self.maximum_prediction_age_s = float(maximum_prediction_age_s)
        self._lock = threading.Lock()
        self._closing = threading.Event()
        self._thread: threading.Thread | None = None
        self._baseline_jpeg: bytes | None = None
        self._max_jpeg: bytes | None = None
        self._tracks: dict[str, TrackState] = {}
        self._raw_detections: tuple[RemoteDetection, ...] = ()
        self._frames = 0
        self._source_frame_id: int | None = None
        self._source_camera_fps: float | None = None
        self._source_inference_ms: float | None = None
        self._source_frame_age_s: float | None = None
        self._last_frame_at: float | None = None
        self._last_source_status_at: float | None = None
        self._error = ""
        self._detector_updates = 0
        self._bridged_track_frames = 0
        self._expired_tracks = 0
        self._rejected_measurements = 0
        self._raw_steps_px: deque[float] = deque(maxlen=180)
        self._fused_steps_px: deque[float] = deque(maxlen=180)

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run,
            name="collie-max-vision",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._closing.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def jpeg(self, view: str) -> bytes | None:
        with self._lock:
            if view == "baseline":
                return self._baseline_jpeg
            return self._max_jpeg

    def status(self) -> dict[str, object]:
        now = time.monotonic()
        with self._lock:
            frame_age_s = (
                None if self._last_frame_at is None else now - self._last_frame_at
            )
            source_status_age_s = (
                None
                if self._last_source_status_at is None
                else now - self._last_source_status_at
            )
            raw_jitter = rms(self._raw_steps_px)
            fused_jitter = rms(self._fused_steps_px)
            reduction = (
                None
                if raw_jitter is None
                or fused_jitter is None
                or raw_jitter <= 1e-6
                else 100.0 * (1.0 - fused_jitter / raw_jitter)
            )
            tracks = [
                {
                    "label": track.label,
                    "confidence": round(track.confidence, 4),
                    "bbox_xywh": tuple(round(value, 2) for value in track.bbox_xywh),
                    "velocity_xywh_s": tuple(
                        round(value, 2) for value in track.velocity_xywh_s
                    ),
                    "verified_age_s": round(now - track.last_verified_at, 3),
                    "source": track.measurement_source,
                    "alpha": round(track.alpha, 3),
                    "center_step_px": round(track.center_step_px, 2),
                }
                for track in sorted(self._tracks.values(), key=lambda item: item.label)
            ]
            return {
                "ok": (
                    frame_age_s is not None
                    and frame_age_s < 1.0
                    and not self._error
                ),
                "service": "collie-max-vision",
                "version": "0.1.0",
                "control_authority": "none_read_only",
                "motion_enabled": False,
                "source_url": self.source_url,
                "source_camera_fps": self._source_camera_fps,
                "source_inference_ms": self._source_inference_ms,
                "source_frame_age_s": self._source_frame_age_s,
                "source_frame_id": self._source_frame_id,
                "frames": self._frames,
                "frame_age_s": (
                    None if frame_age_s is None else round(frame_age_s, 3)
                ),
                "source_status_age_s": (
                    None
                    if source_status_age_s is None
                    else round(source_status_age_s, 3)
                ),
                "baseline": {
                    "backend": "yolo_tensorrt_cuda",
                    "detections": [
                        {
                            "label": item.label,
                            "confidence": item.confidence,
                            "bbox_xyxy": item.bbox_xyxy,
                        }
                        for item in self._raw_detections
                    ],
                },
                "max_vision": {
                    "backend": "max_mojo_temporal_fusion",
                    "tracks": tracks,
                    "maximum_prediction_age_s": self.maximum_prediction_age_s,
                    "detector_updates": self._detector_updates,
                    "bridged_track_frames": self._bridged_track_frames,
                    "expired_tracks": self._expired_tracks,
                    "rejected_measurements": self._rejected_measurements,
                },
                "quality": {
                    "samples": min(len(self._raw_steps_px), len(self._fused_steps_px)),
                    "raw_box_step_rms_px": (
                        None if raw_jitter is None else round(raw_jitter, 2)
                    ),
                    "max_box_step_rms_px": (
                        None if fused_jitter is None else round(fused_jitter, 2)
                    ),
                    "box_step_reduction_percent": (
                        None if reduction is None else round(reduction, 1)
                    ),
                    "description": (
                        "RMS center displacement on matched fresh detector updates; "
                        "lower means less visible box motion, not higher detector accuracy"
                    ),
                },
                "max_mojo": self.fusion.status(),
                "error": self._error,
            }

    def process_observation(
        self,
        frame: np.ndarray[Any, Any],
        source_status: dict[str, object],
        *,
        now: float | None = None,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Process one source observation; exposed for deterministic tests."""

        with self._lock:
            return self._process_observation_locked(
                frame,
                source_status,
                now=now,
            )

    def _process_observation_locked(
        self,
        frame: np.ndarray[Any, Any],
        source_status: dict[str, object],
        *,
        now: float | None,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        observed_at = time.monotonic() if now is None else float(now)
        detections = parse_detections(source_status)
        best_by_label = best_detection_per_label(detections)
        produce = source_status.get("produce")
        produce_payload = produce if isinstance(produce, dict) else {}
        source_frame_id_value = produce_payload.get("frame_id")
        source_frame_id = (
            int(source_frame_id_value)
            if isinstance(source_frame_id_value, int)
            else None
        )
        fresh_detector_cycle = (
            source_frame_id is not None and source_frame_id != self._source_frame_id
        )
        if fresh_detector_cycle:
            self._detector_updates += 1

        candidate_labels = list(self._tracks)
        for detection in detections:
            if detection.label not in candidate_labels:
                candidate_labels.append(detection.label)
        candidate_labels = candidate_labels[:3]

        pending: list[
            tuple[
                str,
                TrackState,
                RemoteDetection | None,
                bool,
                tuple[float, float] | None,
                tuple[float, float] | None,
            ]
        ] = []
        requests: list[TemporalFusionRequest] = []
        for label in candidate_labels:
            detection = best_by_label.get(label)
            track = self._tracks.get(label)
            if track is None:
                if detection is None:
                    continue
                measurement = tuple(float(value) for value in detection.bbox_xywh)
                track = TrackState(
                    label=label,
                    bbox_xywh=measurement,
                    velocity_xywh_s=(0.0, 0.0, 0.0, 0.0),
                    confidence=detection.confidence,
                    last_verified_at=observed_at,
                    last_update_at=max(0.0, observed_at - 1.0 / self.loop_hz),
                    last_detector_frame_id=None,
                )
                self._tracks[label] = track

            measurement_is_fresh = (
                detection is not None
                and (
                    track.last_detector_frame_id is None
                    or (
                        fresh_detector_cycle
                        and source_frame_id != track.last_detector_frame_id
                    )
                )
            )
            verified_age_s = observed_at - track.last_verified_at
            if not measurement_is_fresh and verified_age_s > self.maximum_prediction_age_s:
                self._tracks.pop(label, None)
                self._expired_tracks += 1
                continue

            measurement_bbox = (
                track.bbox_xywh
                if detection is None
                else tuple(float(value) for value in detection.bbox_xywh)
            )
            previous_raw_center = track.last_raw_center
            previous_fused_center = track.last_fused_detection_center
            requests.append(
                TemporalFusionRequest(
                    previous_bbox_xywh=track.bbox_xywh,
                    previous_velocity_xywh_s=track.velocity_xywh_s,
                    measurement_bbox_xywh=measurement_bbox,
                    frame_width=int(frame.shape[1]),
                    frame_height=int(frame.shape[0]),
                    delta_s=max(0.005, observed_at - track.last_update_at),
                    confidence=(
                        track.confidence
                        if detection is None
                        else detection.confidence
                    ),
                    measurement_valid=measurement_is_fresh,
                )
            )
            pending.append(
                (
                    label,
                    track,
                    detection,
                    measurement_is_fresh,
                    previous_raw_center,
                    previous_fused_center,
                )
            )

        results = self.fusion.process(requests) if requests else []
        for item, result in zip(pending, results, strict=True):
            (
                label,
                track,
                detection,
                measurement_is_fresh,
                previous_raw_center,
                previous_fused_center,
            ) = item
            if not result.valid:
                self._tracks.pop(label, None)
                self._expired_tracks += 1
                continue
            track.bbox_xywh = result.bbox_xywh
            track.velocity_xywh_s = result.velocity_xywh_s
            track.last_update_at = observed_at
            track.alpha = result.alpha
            track.center_step_px = result.center_step_px

            if measurement_is_fresh and detection is not None:
                if not result.measurement_used:
                    self._rejected_measurements += 1
                    track.measurement_source = "max_prediction_rejected_measurement"
                    continue
                raw_center = center_xy(
                    tuple(float(value) for value in detection.bbox_xywh)
                )
                fused_center = center_xy(result.bbox_xywh)
                if previous_raw_center is not None:
                    self._raw_steps_px.append(
                        math.dist(previous_raw_center, raw_center)
                    )
                if previous_fused_center is not None:
                    self._fused_steps_px.append(
                        math.dist(previous_fused_center, fused_center)
                    )
                track.last_raw_center = raw_center
                track.last_fused_detection_center = fused_center
                track.last_verified_at = observed_at
                track.last_detector_frame_id = source_frame_id
                track.confidence = detection.confidence
                track.measurement_source = "fresh_yolo_fused_by_max"
            else:
                self._bridged_track_frames += 1
                track.measurement_source = "bounded_max_prediction"

        baseline = frame.copy()
        max_view = frame.copy()
        for detection in detections:
            _draw_detection(baseline, detection)
        for track in self._tracks.values():
            _draw_track(
                max_view,
                track,
                verified_age_s=max(0.0, observed_at - track.last_verified_at),
            )
        _draw_footer(
            baseline,
            "RAW YOLO / TENSORRT",
            (63, 181, 255),
        )
        _draw_footer(
            max_view,
            "MAX + MOJO TEMPORAL FUSION",
            (107, 242, 146),
        )
        self._raw_detections = detections
        self._source_frame_id = source_frame_id
        return baseline, max_view

    def _run(self) -> None:
        period_s = 1.0 / self.loop_hz
        while not self._closing.is_set():
            started = time.monotonic()
            try:
                source_status = self._fetch_json("/api/status")
                frame = self._fetch_frame("/camera-raw.jpg")
                baseline, max_view = self.process_observation(frame, source_status)
                baseline_ok, baseline_jpeg = cv2.imencode(
                    ".jpg",
                    baseline,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 84],
                )
                max_ok, max_jpeg = cv2.imencode(
                    ".jpg",
                    max_view,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 84],
                )
                if not baseline_ok or not max_ok:
                    raise RuntimeError("could not encode MAX comparison frames")
                produce = source_status.get("produce")
                produce_payload = produce if isinstance(produce, dict) else {}
                with self._lock:
                    self._baseline_jpeg = baseline_jpeg.tobytes()
                    self._max_jpeg = max_jpeg.tobytes()
                    self._frames += 1
                    self._last_frame_at = time.monotonic()
                    self._last_source_status_at = time.monotonic()
                    camera_fps = source_status.get("camera_fps")
                    self._source_camera_fps = (
                        float(camera_fps)
                        if isinstance(camera_fps, (int, float))
                        else None
                    )
                    inference_ms = produce_payload.get("inference_ms")
                    self._source_inference_ms = (
                        float(inference_ms)
                        if isinstance(inference_ms, (int, float))
                        else None
                    )
                    source_frame_age = source_status.get("frame_age_s")
                    self._source_frame_age_s = (
                        float(source_frame_age)
                        if isinstance(source_frame_age, (int, float))
                        else None
                    )
                    self._error = ""
            except Exception as exc:  # noqa: BLE001 - service reports and retries
                with self._lock:
                    self._error = f"{type(exc).__name__}: {exc}"
            remaining = period_s - (time.monotonic() - started)
            if remaining > 0.0:
                self._closing.wait(remaining)

    def _fetch_json(self, path: str) -> dict[str, object]:
        request = urllib.request.Request(
            self.source_url + path,
            headers={"Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=1.2) as response:
                payload = json.load(response)
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"source status unavailable: {exc}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("source status was not an object")
        return payload

    def _fetch_frame(self, path: str) -> np.ndarray[Any, Any]:
        try:
            with urllib.request.urlopen(
                self.source_url + path,
                timeout=1.2,
            ) as response:
                content = response.read()
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeError(f"source camera unavailable: {exc}") from exc
        frame = cv2.imdecode(
            np.frombuffer(content, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if frame is None:
            raise RuntimeError("source camera returned an invalid JPEG")
        return frame


def _draw_detection(frame: np.ndarray[Any, Any], detection: RemoteDetection) -> None:
    color = CLASS_COLORS.get(detection.label.casefold(), DEFAULT_COLOR)
    x1, y1, x2, y2 = detection.bbox_xyxy
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
    _draw_label(
        frame,
        f"{detection.label.upper()}  {detection.confidence * 100:.0f}%",
        x1,
        y1,
        color,
    )


def _draw_track(
    frame: np.ndarray[Any, Any],
    track: TrackState,
    *,
    verified_age_s: float,
) -> None:
    color = CLASS_COLORS.get(track.label.casefold(), DEFAULT_COLOR)
    x, y, width, height = (
        int(round(value)) for value in track.bbox_xywh
    )
    predicted = track.measurement_source.startswith("bounded")
    if predicted:
        _draw_dashed_rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            color,
            thickness=3,
        )
        detail = f"MAX BRIDGE {verified_age_s * 1000:.0f}ms"
    else:
        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            color,
            4,
            cv2.LINE_AA,
        )
        detail = f"MAX FUSED  {track.confidence * 100:.0f}%"
    _draw_label(
        frame,
        f"{track.label.upper()}  {detail}",
        x,
        y,
        color,
    )


def _draw_label(
    frame: np.ndarray[Any, Any],
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.62
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        font,
        scale,
        thickness,
    )
    top = max(0, y - text_height - baseline - 14)
    left = max(0, min(frame.shape[1] - text_width - 20, x))
    cv2.rectangle(
        frame,
        (left, top),
        (left + text_width + 18, top + text_height + baseline + 12),
        (5, 12, 10),
        -1,
    )
    cv2.rectangle(
        frame,
        (left, top),
        (left + 5, top + text_height + baseline + 12),
        color,
        -1,
    )
    cv2.putText(
        frame,
        text,
        (left + 12, top + text_height + 5),
        font,
        scale,
        (244, 255, 248),
        thickness,
        cv2.LINE_AA,
    )


def _draw_dashed_rectangle(
    frame: np.ndarray[Any, Any],
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color: tuple[int, int, int],
    *,
    thickness: int,
    dash_px: int = 12,
) -> None:
    x1, y1 = top_left
    x2, y2 = bottom_right
    for start in range(x1, x2, dash_px * 2):
        cv2.line(
            frame,
            (start, y1),
            (min(start + dash_px, x2), y1),
            color,
            thickness,
            cv2.LINE_AA,
        )
        cv2.line(
            frame,
            (start, y2),
            (min(start + dash_px, x2), y2),
            color,
            thickness,
            cv2.LINE_AA,
        )
    for start in range(y1, y2, dash_px * 2):
        cv2.line(
            frame,
            (x1, start),
            (x1, min(start + dash_px, y2)),
            color,
            thickness,
            cv2.LINE_AA,
        )
        cv2.line(
            frame,
            (x2, start),
            (x2, min(start + dash_px, y2)),
            color,
            thickness,
            cv2.LINE_AA,
        )


def _draw_footer(
    frame: np.ndarray[Any, Any],
    text: str,
    color: tuple[int, int, int],
) -> None:
    height, width = frame.shape[:2]
    cv2.rectangle(frame, (0, height - 50), (width, height), (4, 10, 8), -1)
    cv2.rectangle(frame, (0, height - 50), (8, height), color, -1)
    cv2.putText(
        frame,
        text,
        (24, height - 18),
        cv2.FONT_HERSHEY_DUPLEX,
        0.7,
        (238, 255, 245),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "READ ONLY",
        (max(24, width - 150), height - 18),
        cv2.FONT_HERSHEY_DUPLEX,
        0.58,
        color,
        1,
        cv2.LINE_AA,
    )


def create_app(runtime: MaxVisionRuntime) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        runtime.start()
        try:
            yield
        finally:
            runtime.close()

    app = FastAPI(
        title="Collie MAX vision fork",
        version="1",
        lifespan=lifespan,
    )

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return UI_PATH.read_text(encoding="utf-8")

    @app.get("/api/status")
    def status() -> dict[str, object]:
        return runtime.status()

    def camera_response(view: str) -> Response:
        jpeg = runtime.jpeg(view)
        if jpeg is None:
            raise HTTPException(status_code=503, detail="no MAX vision frame yet")
        return Response(
            jpeg,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/camera/baseline.jpg")
    def baseline_camera() -> Response:
        return camera_response("baseline")

    @app.get("/camera/max.jpg")
    def max_camera() -> Response:
        return camera_response("max")

    @app.get("/health")
    def health() -> dict[str, object]:
        status_payload = runtime.status()
        return {
            "ok": status_payload["ok"],
            "service": status_payload["service"],
            "control_authority": status_payload["control_authority"],
            "motion_enabled": status_payload["motion_enabled"],
        }

    return app


def main() -> None:
    device = os.environ.get("COLLIE_MAX_VISION_DEVICE", "cpu")
    fusion = MaxMojoTemporalFusion(device=device)
    runtime = MaxVisionRuntime(
        source_url=os.environ.get(
            "COLLIE_MAX_VISION_SOURCE_URL",
            "http://127.0.0.1:8096",
        ),
        fusion=fusion,
        loop_hz=float(os.environ.get("COLLIE_MAX_VISION_HZ", "12")),
        maximum_prediction_age_s=float(
            os.environ.get("COLLIE_MAX_PREDICTION_AGE_S", "0.45")
        ),
    )
    uvicorn.run(
        create_app(runtime),
        host=os.environ.get("COLLIE_BIND", "0.0.0.0"),
        port=int(os.environ.get("COLLIE_MAX_VISION_PORT", "8107")),
        log_level="info",
    )


if __name__ == "__main__":
    main()
