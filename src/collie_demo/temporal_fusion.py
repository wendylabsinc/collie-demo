from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

BBoxXYWH = tuple[float, float, float, float]
VelocityXYWH = tuple[float, float, float, float]
MAX_TRACKS = 3
RECORD_WIDTH = 18
OUTPUT_WIDTH = 12


@dataclass(frozen=True, slots=True)
class TemporalFusionConfig:
    minimum_measurement_alpha: float = 0.42
    maximum_measurement_alpha: float = 0.82
    velocity_momentum: float = 0.84
    prediction_decay: float = 0.78
    maximum_center_step_fraction: float = 0.22
    minimum_box_size_px: float = 6.0

    def __post_init__(self) -> None:
        if not 0.0 < self.minimum_measurement_alpha <= 1.0:
            raise ValueError("minimum measurement alpha must be in (0, 1]")
        if not self.minimum_measurement_alpha <= self.maximum_measurement_alpha <= 1.0:
            raise ValueError("maximum measurement alpha must be in [minimum, 1]")
        if not 0.0 <= self.velocity_momentum < 1.0:
            raise ValueError("velocity momentum must be in [0, 1)")
        if not 0.0 <= self.prediction_decay <= 1.0:
            raise ValueError("prediction decay must be in [0, 1]")
        if not 0.0 < self.maximum_center_step_fraction <= 1.0:
            raise ValueError("maximum center step fraction must be in (0, 1]")
        if self.minimum_box_size_px <= 0.0:
            raise ValueError("minimum box size must be positive")

    def array(self) -> NDArray[np.float32]:
        return np.asarray(
            [
                self.minimum_measurement_alpha,
                self.maximum_measurement_alpha,
                self.velocity_momentum,
                self.prediction_decay,
                self.maximum_center_step_fraction,
                self.minimum_box_size_px,
            ],
            dtype=np.float32,
        )


@dataclass(frozen=True, slots=True)
class TemporalFusionRequest:
    previous_bbox_xywh: BBoxXYWH
    previous_velocity_xywh_s: VelocityXYWH
    measurement_bbox_xywh: BBoxXYWH
    frame_width: int
    frame_height: int
    delta_s: float
    confidence: float
    measurement_valid: bool
    active: bool = True

    def record(self) -> NDArray[np.float32]:
        return np.asarray(
            (
                *self.previous_bbox_xywh,
                *self.previous_velocity_xywh_s,
                *self.measurement_bbox_xywh,
                float(self.frame_width),
                float(self.frame_height),
                float(self.delta_s),
                float(self.confidence),
                1.0 if self.measurement_valid else 0.0,
                1.0 if self.active else 0.0,
            ),
            dtype=np.float32,
        )


@dataclass(frozen=True, slots=True)
class TemporalFusionResult:
    bbox_xywh: BBoxXYWH
    velocity_xywh_s: VelocityXYWH
    valid: bool
    measurement_used: bool
    alpha: float
    center_step_px: float

    @classmethod
    def from_vector(cls, vector: Sequence[float]) -> TemporalFusionResult:
        if len(vector) != OUTPUT_WIDTH:
            raise ValueError(f"temporal fusion output must have {OUTPUT_WIDTH} values")
        return cls(
            bbox_xywh=tuple(float(value) for value in vector[:4]),
            velocity_xywh_s=tuple(float(value) for value in vector[4:8]),
            valid=float(vector[8]) >= 0.5,
            measurement_used=float(vector[9]) >= 0.5,
            alpha=float(vector[10]),
            center_step_px=float(vector[11]),
        )

    def vector(self) -> NDArray[np.float32]:
        return np.asarray(
            (
                *self.bbox_xywh,
                *self.velocity_xywh_s,
                1.0 if self.valid else 0.0,
                1.0 if self.measurement_used else 0.0,
                self.alpha,
                self.center_step_px,
            ),
            dtype=np.float32,
        )


class TemporalFusionBackend(Protocol):
    name: str

    def process(
        self, requests: Sequence[TemporalFusionRequest]
    ) -> list[TemporalFusionResult]: ...

    def status(self) -> dict[str, object]: ...


class ReferenceTemporalFusion:
    """Scalar reference for deterministic parity tests and safe fallback."""

    name = "python_temporal_reference"

    def __init__(self, config: TemporalFusionConfig | None = None) -> None:
        self.config = config or TemporalFusionConfig()
        self._executions = 0
        self._last_execution_ms: float | None = None
        self._total_execution_ms = 0.0

    def process(
        self, requests: Sequence[TemporalFusionRequest]
    ) -> list[TemporalFusionResult]:
        started = time.perf_counter()
        results = [self._process_one(request) for request in requests]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._executions += 1
        self._last_execution_ms = elapsed_ms
        self._total_execution_ms += elapsed_ms
        return results

    def _process_one(self, request: TemporalFusionRequest) -> TemporalFusionResult:
        record = request.record()
        if not bool(np.all(np.isfinite(record))) or not request.active:
            return _invalid_result()

        frame_width = float(request.frame_width)
        frame_height = float(request.frame_height)
        if frame_width <= 0.0 or frame_height <= 0.0:
            return _invalid_result()

        delta_s = min(0.25, max(0.005, float(request.delta_s)))
        previous = np.asarray(request.previous_bbox_xywh, dtype=np.float32)
        velocity = np.asarray(request.previous_velocity_xywh_s, dtype=np.float32)
        measurement = np.asarray(request.measurement_bbox_xywh, dtype=np.float32)
        prediction = previous + velocity * np.float32(delta_s)

        prediction_center = _center(prediction)
        measurement_center = _center(measurement)
        center_step = float(np.linalg.norm(measurement_center - prediction_center))
        frame_diagonal = math.hypot(frame_width, frame_height)
        measurement_plausible = (
            request.measurement_valid
            and measurement[2] >= self.config.minimum_box_size_px
            and measurement[3] >= self.config.minimum_box_size_px
            and center_step
            <= frame_diagonal * self.config.maximum_center_step_fraction
        )

        if measurement_plausible:
            confidence = min(1.0, max(0.0, float(request.confidence)))
            alpha = self.config.minimum_measurement_alpha + confidence * (
                self.config.maximum_measurement_alpha
                - self.config.minimum_measurement_alpha
            )
            fused = prediction + np.float32(alpha) * (measurement - prediction)
            observed_velocity = (measurement - previous) / np.float32(delta_s)
            next_velocity = (
                np.float32(self.config.velocity_momentum) * velocity
                + np.float32(1.0 - self.config.velocity_momentum)
                * observed_velocity
            )
            measurement_used = True
        else:
            alpha = 0.0
            fused = prediction
            next_velocity = velocity * np.float32(self.config.prediction_decay)
            measurement_used = False
            center_step = float(
                np.linalg.norm(_center(prediction) - _center(previous))
            )

        clipped = _clip_xywh(fused, frame_width, frame_height)
        valid = (
            clipped[2] >= self.config.minimum_box_size_px
            and clipped[3] >= self.config.minimum_box_size_px
        )
        return TemporalFusionResult(
            bbox_xywh=tuple(float(value) for value in clipped),
            velocity_xywh_s=tuple(float(value) for value in next_velocity),
            valid=valid,
            measurement_used=measurement_used,
            alpha=float(alpha),
            center_step_px=center_step,
        )

    def status(self) -> dict[str, object]:
        return _timing_status(
            backend=self.name,
            device="cpu",
            compile_ms=0.0,
            executions=self._executions,
            last_ms=self._last_execution_ms,
            total_ms=self._total_execution_ms,
        )


class MaxMojoTemporalFusion:
    """Batched MAX Graph wrapper around the Collie Mojo temporal filter."""

    name = "max_mojo_temporal_fusion"

    def __init__(
        self,
        *,
        device: str = "auto",
        config: TemporalFusionConfig | None = None,
        kernels_path: Path | None = None,
    ) -> None:
        try:
            from max.driver import CPU, Accelerator, Buffer, accelerator_count
            from max.dtype import DType
            from max.engine import InferenceSession
            from max.graph import DeviceRef, Graph, TensorType, ops
        except ImportError as exc:
            raise RuntimeError(
                "MAX/Mojo is not installed; install Modular before starting "
                "the MAX vision fork"
            ) from exc

        normalized_device = device.casefold().strip()
        if normalized_device not in {"auto", "cpu", "accelerator", "gpu"}:
            raise ValueError("temporal fusion device must be auto, cpu, or accelerator")
        accelerator_total = int(accelerator_count())
        if normalized_device == "cpu":
            resolved_device = CPU()
        elif normalized_device in {"accelerator", "gpu"}:
            if accelerator_total == 0:
                raise RuntimeError("no MAX accelerator is available")
            resolved_device = Accelerator()
        else:
            resolved_device = Accelerator() if accelerator_total else CPU()

        self.config = config or TemporalFusionConfig()
        extension_path = (
            kernels_path
            if kernels_path is not None
            else Path(__file__).parent / "mojo" / "temporal_fusion"
        ).resolve()
        if not extension_path.is_dir():
            raise FileNotFoundError(
                f"Mojo temporal fusion kernels not found: {extension_path}"
            )

        device_ref = DeviceRef.from_device(resolved_device)
        record_type = TensorType(
            DType.float32,
            shape=[MAX_TRACKS, RECORD_WIDTH],
            device=device_ref,
        )
        config_type = TensorType(DType.float32, shape=[6], device=device_ref)
        output_type = TensorType(
            DType.float32,
            shape=[MAX_TRACKS, OUTPUT_WIDTH],
            device=device_ref,
        )
        with Graph(
            "collie_temporal_fusion",
            input_types=[record_type, config_type],
            custom_extensions=[extension_path],
        ) as graph:
            results = ops.custom(
                name="collie_temporal_fusion",
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
        self._last_execution_ms: float | None = None
        self._total_execution_ms = 0.0

    def process(
        self, requests: Sequence[TemporalFusionRequest]
    ) -> list[TemporalFusionResult]:
        if len(requests) > MAX_TRACKS:
            raise ValueError(f"temporal fusion supports at most {MAX_TRACKS} tracks")
        records = np.zeros((MAX_TRACKS, RECORD_WIDTH), dtype=np.float32)
        for index, request in enumerate(requests):
            records[index] = request.record()

        Buffer = self._buffer_type
        record_buffer = Buffer.from_numpy(records).to(self._device)
        config_buffer = Buffer.from_numpy(self.config.array()).to(self._device)
        started = time.perf_counter()
        output = self._model.execute(record_buffer, config_buffer)[0]
        if not isinstance(output, Buffer):
            raise TypeError("MAX temporal fusion returned a non-buffer")
        values = output.to(self._cpu).to_numpy()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._executions += 1
        self._last_execution_ms = elapsed_ms
        self._total_execution_ms += elapsed_ms
        if values.shape != (MAX_TRACKS, OUTPUT_WIDTH):
            raise RuntimeError(
                "MAX temporal fusion returned shape "
                f"{values.shape}; expected {(MAX_TRACKS, OUTPUT_WIDTH)}"
            )
        return [
            TemporalFusionResult.from_vector(values[index])
            for index in range(len(requests))
        ]

    def status(self) -> dict[str, object]:
        return _timing_status(
            backend=self.name,
            device=self._device_name,
            compile_ms=self._compile_ms,
            executions=self._executions,
            last_ms=self._last_execution_ms,
            total_ms=self._total_execution_ms,
        )


def compare_temporal_results(
    reference: Sequence[TemporalFusionResult],
    candidate: Sequence[TemporalFusionResult],
    *,
    tolerance: float = 1e-3,
) -> tuple[bool, float]:
    if len(reference) != len(candidate):
        return False, float("inf")
    maximum_error = 0.0
    for expected, actual in zip(reference, candidate, strict=True):
        expected_values = expected.vector()
        actual_values = actual.vector()
        finite = np.isfinite(expected_values) & np.isfinite(actual_values)
        if bool(np.any(finite)):
            maximum_error = max(
                maximum_error,
                float(np.max(np.abs(expected_values[finite] - actual_values[finite]))),
            )
    return maximum_error <= tolerance, maximum_error


def _center(bbox: NDArray[np.float32]) -> NDArray[np.float32]:
    return np.asarray(
        (bbox[0] + bbox[2] * 0.5, bbox[1] + bbox[3] * 0.5),
        dtype=np.float32,
    )


def _clip_xywh(
    bbox: NDArray[np.float32],
    frame_width: float,
    frame_height: float,
) -> NDArray[np.float32]:
    left = max(0.0, min(frame_width, float(bbox[0])))
    top = max(0.0, min(frame_height, float(bbox[1])))
    right = max(0.0, min(frame_width, float(bbox[0] + bbox[2])))
    bottom = max(0.0, min(frame_height, float(bbox[1] + bbox[3])))
    return np.asarray(
        (left, top, max(0.0, right - left), max(0.0, bottom - top)),
        dtype=np.float32,
    )


def _invalid_result() -> TemporalFusionResult:
    return TemporalFusionResult(
        bbox_xywh=(0.0, 0.0, 0.0, 0.0),
        velocity_xywh_s=(0.0, 0.0, 0.0, 0.0),
        valid=False,
        measurement_used=False,
        alpha=0.0,
        center_step_px=0.0,
    )


def _timing_status(
    *,
    backend: str,
    device: str,
    compile_ms: float,
    executions: int,
    last_ms: float | None,
    total_ms: float,
) -> dict[str, object]:
    return {
        "backend": backend,
        "device": device,
        "compile_ms": round(compile_ms, 3),
        "executions": executions,
        "last_execution_ms": None if last_ms is None else round(last_ms, 3),
        "mean_execution_ms": (
            None if executions == 0 else round(total_ms / executions, 3)
        ),
    }
