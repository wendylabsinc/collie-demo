from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import threading
import time
from typing import Protocol
import urllib.request

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
import numpy as np
from numpy.typing import NDArray
import uvicorn

from .fruit import FruitDetection, FruitDetector, annotate_fruits
from .fruit_webcam import DEFAULT_MODEL


class FrameSource(Protocol):
    description: str

    def read(self) -> NDArray[np.uint8]: ...
    def close(self) -> None: ...


class OpenCVCameraSource:
    def __init__(self, camera_index: int) -> None:
        self.description = f"camera:{camera_index}"
        self.camera = cv2.VideoCapture(camera_index)
        if not self.camera.isOpened():
            raise RuntimeError(f"could not open camera index {camera_index}")

    def read(self) -> NDArray[np.uint8]:
        ok, frame = self.camera.read()
        if not ok or frame is None:
            raise RuntimeError("camera stopped returning frames")
        return frame

    def close(self) -> None:
        self.camera.release()


class HttpJpegSource:
    def __init__(self, url: str) -> None:
        self.description = url
        self.url = url

    def read(self) -> NDArray[np.uint8]:
        request = urllib.request.Request(
            self.url,
            headers={"Cache-Control": "no-cache", "User-Agent": "collie-fruit-ui/1"},
        )
        with urllib.request.urlopen(request, timeout=5.0) as response:
            payload = response.read()
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("HTTP camera returned an invalid JPEG")
        return image

    def close(self) -> None:
        return


class UnitreeFrameSource:
    """Read the Go2's camera directly through Unitree's public VideoClient."""

    def __init__(self, network_interface: str | None, camera: object | None = None) -> None:
        self.description = "unitree:VideoClient"
        if camera is None:
            from .camera import create_camera
            from .motion import initialize_dds

            initialize_dds(network_interface)
            camera = create_camera()
        self.camera = camera

    def read(self) -> NDArray[np.uint8]:
        frame = self.camera.read()
        return frame.bgr

    def close(self) -> None:
        return


class FruitUiRuntime:
    def __init__(self, detector: FruitDetector, source: FrameSource) -> None:
        self.detector = detector
        self.source = source
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._jpeg: bytes | None = None
        self._detections: list[FruitDetection] = []
        self._frame_count = 0
        self._last_frame_at: float | None = None
        self._inference_ms: float | None = None
        self._error = "waiting for first frame"

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="fruit-ui", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=6.0)
        self.source.close()

    def jpeg(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def status(self) -> dict[str, object]:
        with self._lock:
            now = time.monotonic()
            age = None if self._last_frame_at is None else now - self._last_frame_at
            return {
                "ok": age is not None and age < 3.0 and not self._error,
                "frame_count": self._frame_count,
                "frame_age_s": None if age is None else round(age, 3),
                "inference_ms": self._inference_ms,
                "error": self._error,
                "model_path": str(self.detector.model_path),
                "confidence_threshold": self.detector.confidence,
                "classes": self.detector.names,
                "source": self.source.description,
                "detections": [detection.to_dict() for detection in self._detections],
            }

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self.source.read()
                started = time.monotonic()
                detections = self.detector.detect(frame)
                inference_ms = (time.monotonic() - started) * 1000.0
                annotated = annotate_fruits(frame, detections)
                ok, encoded = cv2.imencode(
                    ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 88]
                )
                if not ok:
                    raise RuntimeError("could not encode annotated frame")
                for detection in detections:
                    print(detection.log_line(), flush=True)
                with self._lock:
                    self._jpeg = encoded.tobytes()
                    self._detections = detections
                    self._frame_count += 1
                    self._last_frame_at = time.monotonic()
                    self._inference_ms = round(inference_ms, 1)
                    self._error = ""
            except Exception as exc:
                with self._lock:
                    self._error = str(exc)
                time.sleep(0.25)


def create_app(runtime: FruitUiRuntime, web_file: Path) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        runtime.start()
        try:
            yield
        finally:
            runtime.close()

    app = FastAPI(title="Local Collie fruit detector", lifespan=lifespan)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(web_file)

    @app.get("/api/status")
    def status() -> dict[str, object]:
        return runtime.status()

    @app.get("/camera.jpg")
    def camera() -> Response:
        jpeg = runtime.jpeg()
        if jpeg is None:
            raise HTTPException(status_code=503, detail="no processed frame yet")
        return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    return app


def _source(value: str, network_interface: str | None = None) -> FrameSource:
    if value == "unitree":
        return UnitreeFrameSource(network_interface)
    return OpenCVCameraSource(int(value)) if value.isdigit() else HttpJpegSource(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local browser UI for YOLO fruit detection")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--source",
        default="0",
        help="camera index, JPEG URL, or 'unitree' for the Go2 VideoClient",
    )
    parser.add_argument(
        "--network-interface",
        default=os.environ.get("GO2_NETWORK_INTERFACE", "").strip() or None,
    )
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8097)
    parser.add_argument(
        "--web-file",
        type=Path,
        default=Path("web/tests/fruit-detector.html"),
    )
    args = parser.parse_args()

    detector = FruitDetector(args.model, confidence=args.confidence)
    source = _source(args.source, args.network_interface)
    print("MODEL_PATH=" + str(detector.model_path))
    print("CLASSES=" + json.dumps(detector.names, sort_keys=True))
    app = create_app(FruitUiRuntime(detector, source), args.web_file.resolve())
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
