from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from dataclasses import dataclass
from math import hypot
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
  <title>Collie vision A/B monitor</title>
  <style>
    :root {
      color-scheme: dark;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif;
      --bg: #06100c;
      --panel: #0b1b14;
      --line: #244737;
      --text: #e8fff2;
      --muted: #9bc8af;
      --yolo: #ffb02e;
      --max: #55f29a;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background:
        radial-gradient(circle at 18% 0%, #123324 0, transparent 32rem),
        var(--bg);
      color: var(--text);
    }
    main { width: min(1480px, calc(100% - 32px)); margin: 24px auto 48px; }
    header {
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 24px;
      margin-bottom: 18px;
    }
    h1 { margin: 0; font-size: clamp(1.7rem, 3vw, 2.8rem); letter-spacing: -0.04em; }
    h2 { margin: 0; font-size: 1.1rem; }
    p { color: var(--muted); }
    .safe {
      color: var(--max);
      border: 1px solid #27784b;
      border-radius: 999px;
      padding: 8px 12px;
      font-size: .78rem;
      font-weight: 850;
      letter-spacing: .08em;
      white-space: nowrap;
    }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .panel {
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: color-mix(in srgb, var(--panel) 92%, transparent);
      box-shadow: 0 18px 70px rgba(0,0,0,.22);
    }
    .panel-head { padding: 16px 18px 13px; border-bottom: 1px solid var(--line); }
    .panel-head p { margin: 5px 0 0; font-size: .88rem; }
    .yolo h2 { color: var(--yolo); }
    .max h2 { color: var(--max); }
    img {
      display: block;
      width: 100%;
      aspect-ratio: 16 / 9;
      object-fit: contain;
      background: #000;
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 1px;
      background: var(--line);
      border-top: 1px solid var(--line);
    }
    .metric { min-height: 84px; padding: 13px 15px; background: var(--panel); }
    .metric span {
      display: block;
      color: var(--muted);
      font-size: .72rem;
      font-weight: 750;
      letter-spacing: .06em;
      text-transform: uppercase;
    }
    .metric strong { display: block; margin-top: 7px; font: 750 1.1rem ui-monospace, monospace; }
    .comparison {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 1px;
      margin-top: 16px;
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: var(--line);
    }
    .comparison .metric { min-height: 78px; }
    details {
      margin-top: 16px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: var(--panel);
    }
    summary { cursor: pointer; padding: 13px 16px; color: var(--muted); }
    pre {
      max-height: 320px;
      overflow: auto;
      margin: 0;
      padding: 0 16px 16px;
      white-space: pre-wrap;
      color: #bde5ce;
      font-size: .78rem;
    }
    .waiting { color: var(--muted); }
    .good { color: var(--max); }
    .warn { color: var(--yolo); }
    @media (max-width: 900px) {
      header { align-items: flex-start; flex-direction: column; }
      .grid { grid-template-columns: 1fr; }
      .comparison { grid-template-columns: repeat(2, 1fr); }
    }
    @media (max-width: 520px) {
      main { width: min(100% - 18px, 1480px); margin-top: 14px; }
      .metrics { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>Vision A/B monitor</h1>
      <p>Same camera frame, two different jobs. YOLO detects; KLT + MAX/Mojo tracks.</p>
    </div>
    <div class="safe">READ ONLY · NO MOTION AUTHORITY</div>
  </header>
  <section class="grid">
    <article class="panel yolo">
      <div class="panel-head">
        <h2>YOLO / TensorRT detector</h2>
        <p>Fresh class, confidence, and bounding box from the existing GPU model.</p>
      </div>
      <img id="yolo-camera" src="/camera/yolo.jpg" alt="YOLO detector view">
      <div class="metrics">
        <div class="metric"><span>Detection</span><strong id="yolo-label">Starting…</strong></div>
        <div class="metric"><span>Confidence</span><strong id="yolo-confidence">—</strong></div>
        <div class="metric"><span>Inference</span><strong id="yolo-latency">—</strong></div>
      </div>
    </article>
    <article class="panel max">
      <div class="panel-head">
        <h2>KLT + MAX/Mojo tracker</h2>
        <p>Confidence-neutral optical flow with MAX/Mojo box postprocessing. Not a detector.</p>
      </div>
      <img id="max-camera" src="/camera/max.jpg" alt="MAX-assisted tracker view">
      <div class="metrics">
        <div class="metric"><span>Track</span><strong id="max-label">Starting…</strong></div>
        <div class="metric"><span>KLT update</span><strong id="klt-latency">—</strong></div>
        <div class="metric"><span>MAX kernel</span><strong id="max-latency">—</strong></div>
      </div>
    </article>
  </section>
  <section class="comparison" aria-label="comparison metrics">
    <div class="metric"><span>Availability</span><strong id="availability">Starting…</strong></div>
    <div class="metric"><span>Box agreement · IoU</span><strong id="iou">—</strong></div>
    <div class="metric"><span>Center difference</span><strong id="center-delta">—</strong></div>
    <div class="metric"><span>MAX device</span><strong id="max-device">—</strong></div>
  </section>
  <details>
    <summary>Raw telemetry</summary>
    <pre id="status">Starting…</pre>
  </details>
</main>
<script>
  const yoloCamera = document.getElementById("yolo-camera");
  const maxCamera = document.getElementById("max-camera");
  const status = document.getElementById("status");
  const show = (id, value, fallback = "—") => {
    document.getElementById(id).textContent =
      value === null || value === undefined || value === "" ? fallback : value;
  };
  const ms = value => value === null || value === undefined ? "—" : `${value.toFixed(1)} ms`;
  setInterval(() => {
    const stamp = Date.now();
    yoloCamera.src = "/camera/yolo.jpg?t=" + stamp;
    maxCamera.src = "/camera/max.jpg?t=" + stamp;
  }, 100);
  setInterval(async () => {
    try {
      const response = await fetch("/api/status", {cache: "no-store"});
      const data = await response.json();
      const yolo = data.yolo || {};
      const tracker = data.tracker || {};
      const trackerMetrics = tracker.metrics || {};
      const max = data.max_mojo || {};
      const comparison = data.comparison || {};
      show("yolo-label", yolo.available ? yolo.label : "No detection");
      show("yolo-confidence", yolo.available ? `${(yolo.confidence * 100).toFixed(1)}%` : "N/A");
      show("yolo-latency", ms(yolo.inference_ms));
      show("max-label", tracker.available ? tracker.label : "Waiting for seed");
      show("klt-latency", ms(trackerMetrics.last_update_ms));
      show("max-latency", ms(max.last_execution_ms));
      show("availability", (comparison.availability || "waiting").replaceAll("_", " "));
      show("iou", comparison.iou === null || comparison.iou === undefined
        ? "N/A" : comparison.iou.toFixed(3));
      show("center-delta", comparison.center_delta_px === null ||
        comparison.center_delta_px === undefined
        ? "N/A" : `${comparison.center_delta_px.toFixed(1)} px`);
      show("max-device", max.device || "Unavailable");
      status.textContent = JSON.stringify(data, null, 2);
    } catch (error) {
      status.textContent = String(error);
    }
  }, 250);
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


def parse_detections(payload: object) -> tuple[RemoteDetection, ...]:
    if not isinstance(payload, dict):
        return ()
    produce = payload.get("produce")
    if not isinstance(produce, dict):
        return ()
    detections = produce.get("detections")
    if not isinstance(detections, list):
        return ()
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
    return tuple(sorted(parsed, key=lambda item: item.confidence, reverse=True))


def parse_best_detection(payload: object) -> RemoteDetection | None:
    return next(iter(parse_detections(payload)), None)


def compare_boxes(
    detection: RemoteDetection | None,
    tracked_bbox_xywh: tuple[float, float, float, float] | None,
    *,
    tracker_label: str | None,
) -> dict[str, object]:
    if detection is None and tracked_bbox_xywh is None:
        return {
            "availability": "neither_available",
            "same_label": None,
            "iou": None,
            "center_delta_px": None,
        }
    if detection is None:
        return {
            "availability": "tracker_only_detector_gap",
            "same_label": None,
            "iou": None,
            "center_delta_px": None,
        }
    if tracked_bbox_xywh is None:
        return {
            "availability": "yolo_only",
            "same_label": None,
            "iou": None,
            "center_delta_px": None,
        }

    x, y, width, height = tracked_bbox_xywh
    tracker_xyxy = (x, y, x + width, y + height)
    yolo_xyxy = tuple(float(value) for value in detection.bbox_xyxy)
    intersection_left = max(yolo_xyxy[0], tracker_xyxy[0])
    intersection_top = max(yolo_xyxy[1], tracker_xyxy[1])
    intersection_right = min(yolo_xyxy[2], tracker_xyxy[2])
    intersection_bottom = min(yolo_xyxy[3], tracker_xyxy[3])
    intersection = max(0.0, intersection_right - intersection_left) * max(
        0.0, intersection_bottom - intersection_top
    )
    yolo_area = (yolo_xyxy[2] - yolo_xyxy[0]) * (
        yolo_xyxy[3] - yolo_xyxy[1]
    )
    tracker_area = width * height
    union = yolo_area + tracker_area - intersection
    iou = 0.0 if union <= 0.0 else intersection / union
    yolo_center = (
        (yolo_xyxy[0] + yolo_xyxy[2]) / 2.0,
        (yolo_xyxy[1] + yolo_xyxy[3]) / 2.0,
    )
    tracker_center = (x + width / 2.0, y + height / 2.0)
    return {
        "availability": "both_available",
        "same_label": detection.label == tracker_label,
        "iou": round(iou, 6),
        "center_delta_px": round(
            hypot(
                yolo_center[0] - tracker_center[0],
                yolo_center[1] - tracker_center[1],
            ),
            3,
        ),
    }


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
        self._yolo_jpeg: bytes | None = None
        self._max_jpeg: bytes | None = None
        self._yolo_detection: RemoteDetection | None = None
        self._yolo_inference_ms: float | None = None
        self._comparison: dict[str, object] = compare_boxes(
            None,
            None,
            tracker_label=None,
        )
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

    def jpeg(self, view: str = "max") -> bytes | None:
        with self._lock:
            if view == "yolo":
                return self._yolo_jpeg
            return self._max_jpeg

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
            tracker_payload: dict[str, object]
            if tracker_status is None:
                tracker_payload = {
                    "available": False,
                    "label": None,
                    "bbox_xywh": None,
                    "backend": "opencv_klt_affine",
                    "metrics": {},
                }
            else:
                tracker_payload = {
                    **tracker_status,
                    "available": True,
                    "label": self._tracker_label,
                }
            yolo = self._yolo_detection
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
                "yolo": {
                    "available": yolo is not None,
                    "label": None if yolo is None else yolo.label,
                    "confidence": None if yolo is None else yolo.confidence,
                    "bbox_xyxy": None if yolo is None else yolo.bbox_xyxy,
                    "inference_ms": self._yolo_inference_ms,
                    "backend": "tensorrt_cuda",
                },
                "tracker": tracker_payload,
                "comparison": dict(self._comparison),
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
                yolo_view, max_view, comparison = self._process_frame(
                    frame,
                    detection,
                )
                yolo_encoded, yolo_jpeg = cv2.imencode(
                    ".jpg",
                    yolo_view,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 82],
                )
                max_encoded, max_jpeg = cv2.imencode(
                    ".jpg",
                    max_view,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 82],
                )
                if not yolo_encoded or not max_encoded:
                    raise RuntimeError("could not encode comparison frames")
                produce = (
                    source_status.get("produce")
                    if isinstance(source_status, dict)
                    else None
                )
                with self._lock:
                    self._yolo_jpeg = yolo_jpeg.tobytes()
                    self._max_jpeg = max_jpeg.tobytes()
                    self._yolo_detection = detection
                    inference_ms = (
                        produce.get("inference_ms")
                        if isinstance(produce, dict)
                        else None
                    )
                    self._yolo_inference_ms = (
                        float(inference_ms)
                        if isinstance(inference_ms, (int, float))
                        else None
                    )
                    self._comparison = comparison
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
    ) -> tuple[
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
        dict[str, object],
    ]:
        yolo_view = frame.copy()
        max_view = frame.copy()
        tracker = self._tracker
        tracker_label = self._tracker_label
        tracked_bbox: tuple[float, float, float, float] | None = None

        if (
            tracker is not None
            and detection is not None
            and tracker_label != detection.label
        ):
            tracker = None
            tracker_label = None

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
            cv2.rectangle(yolo_view, (x1, y1), (x2, y2), (0, 180, 255), 4)
            cv2.putText(
                yolo_view,
                f"YOLO {detection.label} {detection.confidence:.2f}",
                (x1, max(24, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 180, 255),
                3,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                yolo_view,
                "NO YOLO DETECTION",
                (24, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 180, 255),
                3,
                cv2.LINE_AA,
            )
        if tracked_bbox is not None and tracker_label is not None:
            x, y, width, height = (
                int(round(value)) for value in tracked_bbox
            )
            cv2.rectangle(
                max_view,
                (x, y),
                (x + width, y + height),
                (80, 255, 130),
                4,
            )
            cv2.putText(
                max_view,
                f"KLT + MAX/MOJO {tracker_label} (confidence N/A)",
                (max(8, x), max(50, y - 34)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (80, 255, 130),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                max_view,
                "WAITING FOR YOLO SEED",
                (24, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (80, 255, 130),
                3,
                cv2.LINE_AA,
            )
        comparison = compare_boxes(
            detection,
            tracked_bbox,
            tracker_label=tracker_label,
        )
        for view, label in (
            (yolo_view, "YOLO / TENSORRT DETECTOR"),
            (max_view, "KLT TRACKER + MAX/MOJO POSTPROCESS"),
        ):
            cv2.putText(
                view,
                label,
                (20, frame.shape[0] - 54),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.66,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            yolo_view,
            "READ ONLY - NO MOTION AUTHORITY",
            (20, frame.shape[0] - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            max_view,
            "READ ONLY - NO MOTION AUTHORITY",
            (20, frame.shape[0] - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return yolo_view, max_view, comparison

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

    def camera_response(view: str) -> Response:
        jpeg = runtime.jpeg(view)
        if jpeg is None:
            raise HTTPException(status_code=503, detail="no shadow frame yet")
        return Response(
            jpeg,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/camera.jpg")
    def camera() -> Response:
        return camera_response("max")

    @app.get("/camera/yolo.jpg")
    def yolo_camera() -> Response:
        return camera_response("yolo")

    @app.get("/camera/max.jpg")
    def max_camera() -> Response:
        return camera_response("max")

    return app


def main() -> None:
    runtime = ShadowRuntime(
        source_url=os.environ.get(
            "COLLIE_SHADOW_SOURCE_URL",
            "http://127.0.0.1:8096",
        ),
        device=os.environ.get("COLLIE_BOX_POSTPROCESS_DEVICE", "cpu"),
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
