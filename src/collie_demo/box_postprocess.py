from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

BBoxXYWH = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class BoxPostprocessRequest:
    bbox_xywh: BBoxXYWH
    affine_2x3: tuple[float, float, float, float, float, float]
    frame_width: int
    frame_height: int
    minimum_box_size_px: float
    minimum_scale_per_frame: float
    maximum_scale_per_frame: float
    maximum_center_step_fraction: float
    new_box_weight: float

    @classmethod
    def from_affine(
        cls,
        *,
        bbox_xywh: BBoxXYWH,
        affine_2x3: NDArray[np.floating],
        frame_width: int,
        frame_height: int,
        minimum_box_size_px: float,
        minimum_scale_per_frame: float,
        maximum_scale_per_frame: float,
        maximum_center_step_fraction: float,
        new_box_weight: float,
    ) -> BoxPostprocessRequest:
        affine = np.asarray(affine_2x3, dtype=np.float32)
        if affine.shape != (2, 3):
            raise ValueError("affine transform must have shape (2, 3)")
        return cls(
            bbox_xywh=tuple(float(value) for value in bbox_xywh),
            affine_2x3=tuple(float(value) for value in affine.reshape(-1)),
            frame_width=int(frame_width),
            frame_height=int(frame_height),
            minimum_box_size_px=float(minimum_box_size_px),
            minimum_scale_per_frame=float(minimum_scale_per_frame),
            maximum_scale_per_frame=float(maximum_scale_per_frame),
            maximum_center_step_fraction=float(maximum_center_step_fraction),
            new_box_weight=float(new_box_weight),
        )

    def record_array(self) -> NDArray[np.float32]:
        return np.asarray(
            [
                (
                    *self.bbox_xywh,
                    *self.affine_2x3,
                    float(self.frame_width),
                    float(self.frame_height),
                )
            ],
            dtype=np.float32,
        )

    def config_array(self) -> NDArray[np.float32]:
        return np.asarray(
            [
                self.minimum_box_size_px,
                self.minimum_scale_per_frame,
                self.maximum_scale_per_frame,
                self.maximum_center_step_fraction,
                self.new_box_weight,
            ],
            dtype=np.float32,
        )


@dataclass(frozen=True, slots=True)
class BoxPostprocessResult:
    bbox_xywh: BBoxXYWH
    valid: bool
    scale: float
    center_step_px: float
    failure_reason: str = ""

    def numeric_vector(self) -> NDArray[np.float32]:
        return np.asarray(
            (
                *self.bbox_xywh,
                1.0 if self.valid else 0.0,
                self.scale,
                self.center_step_px,
            ),
            dtype=np.float32,
        )


class BoxPostprocessor(Protocol):
    name: str

    def process(self, request: BoxPostprocessRequest) -> BoxPostprocessResult: ...


@dataclass(frozen=True, slots=True)
class BoxPostprocessComparison:
    matches: bool
    maximum_absolute_error: float
    valid_matches: bool


class ReferenceBoxPostprocessor:
    """Trusted scalar implementation used for all control-facing results."""

    name = "python_reference"

    def process(self, request: BoxPostprocessRequest) -> BoxPostprocessResult:
        record = request.record_array()[0]
        config = request.config_array()
        (
            x,
            y,
            width,
            height,
            a00,
            a01,
            a02,
            a10,
            a11,
            a12,
            frame_width,
            frame_height,
        ) = (np.float32(value) for value in record)
        (
            minimum_size,
            minimum_scale,
            maximum_scale,
            maximum_step_fraction,
            new_box_weight,
        ) = (np.float32(value) for value in config)

        values = np.asarray((*record, *config), dtype=np.float32)
        if not bool(np.all(np.isfinite(values))):
            return BoxPostprocessResult(
                request.bbox_xywh,
                False,
                float("nan"),
                float("nan"),
                "non_finite_input",
            )
        if frame_width <= 0.0 or frame_height <= 0.0:
            return BoxPostprocessResult(
                request.bbox_xywh,
                False,
                0.0,
                0.0,
                "invalid_frame_size",
            )

        corners = (
            (x, y),
            (x + width, y),
            (x + width, y + height),
            (x, y + height),
        )
        transformed = tuple(
            (
                a00 * corner_x + a01 * corner_y + a02,
                a10 * corner_x + a11 * corner_y + a12,
            )
            for corner_x, corner_y in corners
        )
        left = min(point[0] for point in transformed)
        top = min(point[1] for point in transformed)
        right = max(point[0] for point in transformed)
        bottom = max(point[1] for point in transformed)
        clipped_left = max(np.float32(0.0), min(frame_width, left))
        clipped_top = max(np.float32(0.0), min(frame_height, top))
        clipped_right = max(np.float32(0.0), min(frame_width, right))
        clipped_bottom = max(np.float32(0.0), min(frame_height, bottom))
        clipped_width = clipped_right - clipped_left
        clipped_height = clipped_bottom - clipped_top

        scale = np.float32(math.hypot(float(a00), float(a10)))
        old_center_x = x + width / np.float32(2.0)
        old_center_y = y + height / np.float32(2.0)
        new_center_x = clipped_left + clipped_width / np.float32(2.0)
        new_center_y = clipped_top + clipped_height / np.float32(2.0)
        center_step = np.float32(
            math.hypot(
                float(new_center_x - old_center_x),
                float(new_center_y - old_center_y),
            )
        )
        frame_diagonal = np.float32(math.hypot(float(frame_width), float(frame_height)))

        failure_reason = ""
        if scale < minimum_scale or scale > maximum_scale:
            failure_reason = "implausible_scale"
        elif clipped_width < minimum_size or clipped_height < minimum_size:
            failure_reason = "box_left_frame"
        elif center_step > frame_diagonal * maximum_step_fraction:
            failure_reason = "implausible_center_step"

        previous_weight = np.float32(1.0) - new_box_weight
        blended = (
            previous_weight * x + new_box_weight * clipped_left,
            previous_weight * y + new_box_weight * clipped_top,
            previous_weight * width + new_box_weight * clipped_width,
            previous_weight * height + new_box_weight * clipped_height,
        )
        return BoxPostprocessResult(
            bbox_xywh=tuple(float(value) for value in blended),
            valid=not failure_reason,
            scale=float(scale),
            center_step_px=float(center_step),
            failure_reason=failure_reason,
        )


class MaxMojoBoxPostprocessor:
    """MAX Graph wrapper around the shadow-only Mojo custom operation."""

    name = "max_mojo"

    def __init__(
        self,
        *,
        device: str = "auto",
        kernels_path: Path | None = None,
    ) -> None:
        try:
            from max.driver import CPU, Accelerator, Buffer, accelerator_count
            from max.dtype import DType
            from max.engine import InferenceSession
            from max.graph import DeviceRef, Graph, TensorType, ops
        except ImportError as exc:
            raise RuntimeError(
                "MAX/Mojo is not installed; install the Modular package from "
                "the official Modular package index"
            ) from exc

        normalized_device = device.casefold().strip()
        if normalized_device not in {"auto", "cpu", "accelerator", "gpu"}:
            raise ValueError("box postprocess device must be auto, cpu, or accelerator")
        available_accelerators = int(accelerator_count())
        if normalized_device == "cpu":
            resolved_device = CPU()
        elif normalized_device in {"accelerator", "gpu"}:
            if available_accelerators == 0:
                raise RuntimeError("no MAX accelerator is available")
            resolved_device = Accelerator()
        else:
            resolved_device = Accelerator() if available_accelerators else CPU()

        extension_path = (
            kernels_path
            if kernels_path is not None
            else Path(__file__).parent / "mojo" / "box_postprocess"
        )
        extension_path = extension_path.resolve()
        if not extension_path.is_dir():
            raise FileNotFoundError(
                f"Mojo box postprocess kernels not found: {extension_path}"
            )

        device_ref = DeviceRef.from_device(resolved_device)
        record_type = TensorType(DType.float32, shape=[1, 12], device=device_ref)
        config_type = TensorType(DType.float32, shape=[5], device=device_ref)
        output_type = TensorType(DType.float32, shape=[1, 7], device=device_ref)
        with Graph(
            "collie_box_postprocess",
            input_types=[record_type, config_type],
            custom_extensions=[extension_path],
        ) as graph:
            results = ops.custom(
                name="collie_box_postprocess",
                device=device_ref,
                values=[graph.inputs[0], graph.inputs[1]],
                out_types=[output_type],
            )
            graph.output(*results)

        compile_started = time.perf_counter()
        self._session = InferenceSession(devices=[resolved_device])
        self._model = self._session.load(graph)
        self._compile_ms = (time.perf_counter() - compile_started) * 1000.0
        self._device = resolved_device
        self._device_name = str(resolved_device)
        self._cpu = CPU()
        self._buffer_type = Buffer
        self._executions = 0
        self._total_execution_ms = 0.0
        self._last_execution_ms: float | None = None

    def process(self, request: BoxPostprocessRequest) -> BoxPostprocessResult:
        Buffer = self._buffer_type
        record = Buffer.from_numpy(request.record_array()).to(self._device)
        config = Buffer.from_numpy(request.config_array()).to(self._device)
        started = time.perf_counter()
        output = self._model.execute(record, config)[0]
        if not isinstance(output, Buffer):
            raise TypeError("MAX box postprocess returned a non-buffer")
        values = output.to(self._cpu).to_numpy().reshape(-1)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._executions += 1
        self._total_execution_ms += elapsed_ms
        self._last_execution_ms = elapsed_ms
        if len(values) != 7:
            raise RuntimeError(
                f"MAX box postprocess returned {len(values)} values; expected 7"
            )
        valid = bool(float(values[4]) >= 0.5)
        return BoxPostprocessResult(
            bbox_xywh=tuple(float(value) for value in values[:4]),
            valid=valid,
            scale=float(values[5]),
            center_step_px=float(values[6]),
            failure_reason="" if valid else "max_mojo_rejected",
        )

    def status(self) -> dict[str, object]:
        return {
            "backend": self.name,
            "device": self._device_name,
            "compile_ms": round(self._compile_ms, 3),
            "executions": self._executions,
            "last_execution_ms": None
            if self._last_execution_ms is None
            else round(self._last_execution_ms, 3),
            "mean_execution_ms": None
            if self._executions == 0
            else round(
                self._total_execution_ms / self._executions,
                3,
            ),
        }


def compare_box_postprocess_results(
    reference: BoxPostprocessResult,
    candidate: BoxPostprocessResult,
    *,
    absolute_tolerance: float = 1e-3,
) -> BoxPostprocessComparison:
    reference_values = reference.numeric_vector()
    candidate_values = candidate.numeric_vector()
    finite_mask = np.isfinite(reference_values) & np.isfinite(candidate_values)
    if bool(np.any(finite_mask)):
        maximum_error = float(
            np.max(
                np.abs(reference_values[finite_mask] - candidate_values[finite_mask])
            )
        )
    else:
        maximum_error = 0.0
    valid_matches = reference.valid == candidate.valid
    return BoxPostprocessComparison(
        matches=valid_matches and maximum_error <= absolute_tolerance,
        maximum_absolute_error=maximum_error,
        valid_matches=valid_matches,
    )


def shadow_box_postprocessor_from_mode(
    mode: str,
    *,
    device: str = "auto",
) -> BoxPostprocessor | None:
    normalized = mode.casefold().strip().replace("-", "_")
    if normalized in {"", "off", "none", "disabled"}:
        return None
    if normalized not in {"max", "mojo", "max_mojo"}:
        raise ValueError("COLLIE_BOX_POSTPROCESS_SHADOW must be off or max_mojo")
    return MaxMojoBoxPostprocessor(device=device)
