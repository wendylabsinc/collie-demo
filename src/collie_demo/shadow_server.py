from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response

from .box_postprocess import MaxMojoBoxPostprocessor
from .box_tracking import BoxTrackerInitializationError, KLTBoxTracker


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Collie box shadow</title>
  <style>
    :root { color-scheme: dark; font-family: ui-monospace, SFMono-Regular, monospace; }
    body { margin: 0; background: #07110d; color: #d9fbe8; }
    main { width: min(1100px, calc(100% - 32px)); margin: 24px auto; }
    h1 { margin: 0 0 8px; }
    .safe { color: #5cff9f; font-weight: 800; }
    img { width: 100%; border: 1px solid #315f48; border-radius: 12px; background: #000; }
    pre { white-space: pre-wrap; background: #0c1e15; padding: 16px; border-radius: 12px; }
  </style>
</head>
<body>
<main>
  <h1>Collie box-tracker shadow</h1>
  <p class="safe">READ ONLY · NO ROBOT MOTION AUTHORITY</p>
  <img id="camera" src="/camera.jpg" alt="shadow tracker camera">
  <pre id="status">Starting…</pre>
</main>
<script>
  const camera = document.getElementById("camera");
  const status = document.getElementById("status");
  setInterval(() => { camera.src = "/camera.jpg?t=" + Date.now(); }, 100);
  setInterval(async () => {
    try {
      const response = await fetch("/api/status", {cache: "no-store"});
      status.textContent = JSON.stringify(await response.json(), null, 2);
    } catch (error) {
      status.textContent = String(error);
    }
  }, 500);
</script>
</body>
</html>
"""


@dataclass(frozen=True, slots=True)
class RemoteDetection:
    label: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]

    @property
    def bbox_xywh(self) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return x1, y1, x2 - x1, y2 - y1


def parse_best_detection(payload: object) -> RemoteDetection | None:
    if not isinstance(payload, dict):
        return None
    produce = payload.get("produce")
    if not isinstance(produce, dict):
        return None
    detections = produce.get("detections")
    if not isinstance(detections, list):
        return None
    parsed: list[RemoteDetection] = []
    for item in detections:
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        confidence = item.get("confidence")
        bbox = item.get("bbox_xyxy")
        if (
            not isinstance(label, str)
            or not isinstance(confidence, (int, float))
            or not isinstance(bbox, (list, tuple))
            or len(bbox) != 4
        ):
            continue
        try:
            coordinates = tuple(int(value) for value in bbox)
        except (TypeError, ValueError):
            continue
        x1, y1, x2, y2 = coordinates
        if x2 <= x1 or y2 <= y1:
            continue
        parsed.append(
            RemoteDetection(
                label=label,
                confidence=float(confidence),
                bbox_xyxy=coordinates,
            )
        )
    return max(parsed, key=lambda item: item.confidence, default=None)


class ShadowRuntime:
    def __init__(
        self,
        *,
        source_url: str,
        device: str,
        loop_hz: float,
    ) -> None:
        self.source_url = source_url.rstrip("/")
        self.device = device
        self.loop_hz = max(1.0, float(loop_hz))
        self._lock = threading.Lock()
        self._closing = threading.Event()
        self._thread: threading.Thread | None = None
        self._postprocessor: MaxMojoBoxPostprocessor | None = None
        self._tracker: KLTBoxTracker | None = None
        self._tracker_label: str | None = None
        self._jpeg: bytes | None = None
        self._started_at = time.monotonic()
        self._last_frame_at: float | None = None
        self._last_source_status_at: float | None = None
        self._frames = 0
        self._source_frame_id: int | None = None
        self._source_camera_fps: float | None = None
        self._error = ""
        self._tracker_initializations = 0
        self._tracker_failures = 0

    def start(self) -> None:
        self._postprocessor = MaxMojoBoxPostprocessor(device=self.device)
        self._thread = threading.Thread(
            target=self._run,
            name="collie-box-shadow",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._closing.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def jpeg(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def status(self) -> dict[str, object]:
        now = time.monotonic()
        with self._lock:
            frame_age = (
                None if self._last_frame_at is None else now - self._last_frame_at
            )
            status_age = (
                None
                if self._last_source_status_at is None
                else now - self._last_source_status_at
            )
            tracker_status = (
                None if self._tracker is None else self._tracker.status()
            )
            return {
                "ok": frame_age is not None and frame_age < 1.0 and not self._error,
                "service": "collie-box-shadow",
                "control_authority": "none_shadow_only",
                "motion_enabled": False,
                "source_url": self.source_url,
                "source_frame_id": self._source_frame_id,
                "source_camera_fps": self._source_camera_fps,
                "frames": self._frames,
                "frame_age_s": None if frame_age is None else round(frame_age, 3),
                "source_status_age_s": (
                    None if status_age is None else round(status_age, 3)
                ),
                "tracker_label": self._tracker_label,
                "tracker_initializations": self._tracker_initializations,
                "tracker_failures": self._tracker_failures,
                "tracker": tracker_status,
                "max_mojo": (
                    None
                    if self._postprocessor is None
                    else self._postprocessor.status()
                ),
                "error": self._error,
            }

    def _run(self) -> None:
        period_s = 1.0 / self.loop_hz
        while not self._closing.is_set():
            started = time.monotonic()
            try:
                source_status = self._fetch_json("/api/status")
                frame = self._fetch_frame("/camera-raw.jpg")
                detection = parse_best_detection(source_status)
                annotated = self._process_frame(frame, detection)
                encoded, jpeg = cv2.imencode(
                    ".jpg",
                    annotated,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 82],
                )
                if not encoded:
                    raise RuntimeError("could not encode shadow frame")
                produce = (
                    source_status.get("produce")
                    if isinstance(source_status, dict)
                    else None
                )
                with self._lock:
                    self._jpeg = jpeg.tobytes()
                    self._frames += 1
                    self._last_frame_at = time.monotonic()
                    self._last_source_status_at = time.monotonic()
                    self._source_frame_id = (
                        produce.get("frame_id")
                        if isinstance(produce, dict)
                        and isinstance(produce.get("frame_id"), int)
                        else None
                    )
                    camera_fps = source_status.get("camera_fps")
                    self._source_camera_fps = (
                        float(camera_fps)
                        if isinstance(camera_fps, (int, float))
                        else None
                    )
                    self._error = ""
            except Exception as exc:  # noqa: BLE001 - service reports and retries
                with self._lock:
                    self._error = f"{type(exc).__name__}: {exc}"
            remaining = period_s - (time.monotonic() - started)
            if remaining > 0:
                self._closing.wait(remaining)

    def _process_frame(
        self,
        frame: np.ndarray[Any, Any],
        detection: RemoteDetection | None,
    ) -> np.ndarray[Any, Any]:
        annotated = frame.copy()
        tracker = self._tracker
        tracker_label = self._tracker_label
        tracked_bbox: tuple[float, float, float, float] | None = None

        if tracker is not None:
            tracker_ok, tracked_bbox = tracker.update(frame)
            if not tracker_ok:
                with self._lock:
                    self._tracker_failures += 1
                tracker = None
                tracker_label = None
                tracked_bbox = None

        if tracker is None and detection is not None:
            try:
                tracker = KLTBoxTracker(
                    frame,
                    detection.bbox_xywh,
                    shadow_postprocessor=self._postprocessor,
                )
                tracker_label = detection.label
                tracked_bbox = tuple(float(value) for value in detection.bbox_xywh)
                with self._lock:
                    self._tracker_initializations += 1
            except BoxTrackerInitializationError:
                tracker = None
                tracker_label = None

        self._tracker = tracker
        self._tracker_label = tracker_label
        if detection is not None:
            x1, y1, x2, y2 = detection.bbox_xyxy
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 180, 255), 2)
            cv2.putText(
                annotated,
                f"YOLO {detection.label} {detection.confidence:.2f}",
                (x1, max(24, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 180, 255),
                2,
                cv2.LINE_AA,
            )
        if tracked_bbox is not None and tracker_label is not None:
            x, y, width, height = (
                int(round(value)) for value in tracked_bbox
            )
            cv2.rectangle(
                annotated,
                (x, y),
                (x + width, y + height),
                (80, 255, 130),
                4,
            )
            cv2.putText(
                annotated,
                f"SHADOW KLT + MAX/MOJO {tracker_label}",
                (max(8, x), max(50, y - 34)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (80, 255, 130),
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            annotated,
            "READ ONLY - NO MOTION AUTHORITY",
            (20, frame.shape[0] - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return annotated

    def _fetch_json(self, path: str) -> dict[str, object]:
        request = urllib.request.Request(
            self.source_url + path,
            headers={"Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=1.0) as response:
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
                timeout=1.0,
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


def create_app(runtime: ShadowRuntime) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        runtime.start()
        try:
            yield
        finally:
            runtime.close()

    app = FastAPI(
        title="Collie box-tracker shadow",
        version="1",
        lifespan=lifespan,
    )

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return INDEX_HTML

    @app.get("/api/status")
    def status() -> dict[str, object]:
        return runtime.status()

    @app.get("/camera.jpg")
    def camera() -> Response:
        jpeg = runtime.jpeg()
        if jpeg is None:
            raise HTTPException(status_code=503, detail="no shadow frame yet")
        return Response(
            jpeg,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    return app


def main() -> None:
    runtime = ShadowRuntime(
        source_url=os.environ.get(
            "COLLIE_SHADOW_SOURCE_URL",
            "http://127.0.0.1:8096",
        ),
        device=os.environ.get("COLLIE_BOX_POSTPROCESS_DEVICE", "accelerator"),
        loop_hz=float(os.environ.get("COLLIE_SHADOW_HZ", "12")),
    )
    uvicorn.run(
        create_app(runtime),
        host=os.environ.get("COLLIE_BIND", "0.0.0.0"),
        port=int(os.environ.get("COLLIE_SHADOW_PORT", "8106")),
        log_level="info",
    )


if __name__ == "__main__":
    main()
