"""Per-round target-class memory for the fruit demonstration."""

from __future__ import annotations

from dataclasses import dataclass
import secrets
import time

import cv2
from numpy.typing import NDArray
import numpy as np


@dataclass(frozen=True, slots=True)
class FruitMemory:
    memory_id: str
    label: str
    created_monotonic_s: float
    reference_jpeg: bytes
    reference_bbox_xyxy: tuple[int, int, int, int]

    @classmethod
    def create(
        cls,
        *,
        label: str,
        reference_jpeg: bytes,
        reference_bbox_xyxy: tuple[int, int, int, int],
    ) -> "FruitMemory":
        normalized = label.casefold().strip()
        if not normalized:
            raise ValueError("fruit class label cannot be empty")
        return cls(
            memory_id=secrets.token_urlsafe(10),
            label=normalized,
            created_monotonic_s=time.monotonic(),
            reference_jpeg=reference_jpeg,
            reference_bbox_xyxy=reference_bbox_xyxy,
        )

    def to_status(self, now: float | None = None) -> dict[str, object]:
        current = time.monotonic() if now is None else now
        return {
            "id": self.memory_id,
            "label": self.label,
            "role": "target_class",
            "age_s": round(max(0.0, current - self.created_monotonic_s), 3),
            "reference_bbox_xyxy": self.reference_bbox_xyxy,
            "has_reference": bool(self.reference_jpeg),
        }


def encode_jpeg(crop: NDArray[np.uint8]) -> bytes:
    ok, encoded = cv2.imencode(
        ".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 94]
    )
    if not ok:
        raise RuntimeError("could not encode fruit-class reference")
    return encoded.tobytes()


def crop_bbox(
    bgr: NDArray[np.uint8], bbox_xyxy: tuple[int, int, int, int]
) -> NDArray[np.uint8]:
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("camera frame must be a BGR image")
    x1, y1, x2, y2 = bbox_xyxy
    height, width = bgr.shape[:2]
    x1 = max(0, min(width - 1, int(x1)))
    y1 = max(0, min(height - 1, int(y1)))
    x2 = max(x1 + 1, min(width, int(x2)))
    y2 = max(y1 + 1, min(height, int(y2)))
    return bgr[y1:y2, x1:x2].copy()
