from __future__ import annotations

from dataclasses import asdict, dataclass
from threading import Lock

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True, init=False)
class CameraFrame:
    frame_id: int
    captured_monotonic_s: float
    _bgr: NDArray[np.uint8] | None
    # Unitree already sends a JPEG. Keeping the original bytes lets the web
    # stream forward frames without avoidable full-resolution decode/encode.
    source_jpeg: bytes | None
    # Opaque producer connection identity. A change invalidates every
    # camera-derived lock even when local frame IDs remain monotonic.
    stream_generation: str | None
    _width: int
    _height: int
    _decode_lock: Lock

    def __init__(
        self,
        frame_id: int,
        captured_monotonic_s: float,
        bgr: NDArray[np.uint8] | None,
        source_jpeg: bytes | None = None,
        width: int | None = None,
        height: int | None = None,
        stream_generation: str | None = None,
    ) -> None:
        if bgr is None and source_jpeg is None:
            raise ValueError("a camera frame requires an image or source JPEG")
        if bgr is not None:
            if bgr.ndim != 3:
                raise ValueError("camera image must have three dimensions")
            height, width = int(bgr.shape[0]), int(bgr.shape[1])
        if width is None or height is None or width <= 0 or height <= 0:
            raise ValueError("camera frame dimensions must be positive")
        self.frame_id = int(frame_id)
        self.captured_monotonic_s = float(captured_monotonic_s)
        self._bgr = bgr
        self.source_jpeg = source_jpeg
        self.stream_generation = (
            None if stream_generation is None else str(stream_generation)
        )
        self._width = int(width)
        self._height = int(height)
        self._decode_lock = Lock()

    @property
    def bgr(self) -> NDArray[np.uint8]:
        """Decode the JPEG only when inference or image analysis needs it."""

        with self._decode_lock:
            if self._bgr is None:
                assert self.source_jpeg is not None
                encoded = np.frombuffer(self.source_jpeg, dtype=np.uint8)
                image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                if image is None or image.ndim != 3:
                    raise RuntimeError("camera returned an invalid JPEG")
                self._bgr = image
            return self._bgr

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height


@dataclass(frozen=True, slots=True)
class TargetObservation:
    frame_id: int
    captured_monotonic_s: float
    bbox_xywh: tuple[int, int, int, int]
    center: tuple[int, int]
    confidence: float | None
    visible_frames: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

@dataclass(frozen=True, slots=True)
class VelocityCommand:
    forward_mps: float = 0.0
    yaw_rps: float = 0.0
    reason: str = "idle"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
