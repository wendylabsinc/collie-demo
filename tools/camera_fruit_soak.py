#!/usr/bin/env python3
"""Read-only camera and fruit-recognition reliability probe for Woof."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

TARGETS = ("apple", "banana", "pear")


@dataclass(frozen=True, slots=True)
class SoakConfig:
    maximum_camera_age_s: float = 0.75
    maximum_detection_age_s: float = 0.75
    maximum_frame_stall_s: float = 1.5
    maximum_status_request_s: float = 1.0
    maximum_generation_changes: int = 0
    expected_fruit: str | None = None
    minimum_recognition_ratio: float = 0.80
    recognition_warmup_s: float = 3.0

    def __post_init__(self) -> None:
        for name in (
            "maximum_camera_age_s",
            "maximum_detection_age_s",
            "maximum_frame_stall_s",
            "maximum_status_request_s",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.maximum_generation_changes < 0:
            raise ValueError("maximum_generation_changes cannot be negative")
        if self.expected_fruit not in {None, *TARGETS}:
            raise ValueError(f"expected_fruit must be one of {TARGETS}")
        if not 0.0 <= self.minimum_recognition_ratio <= 1.0:
            raise ValueError("minimum_recognition_ratio must be between 0 and 1")
        if (
            not math.isfinite(self.recognition_warmup_s)
            or self.recognition_warmup_s < 0
        ):
            raise ValueError("recognition_warmup_s must be non-negative")


@dataclass(slots=True)
class _ProgressMarker:
    value: int
    changed_at_s: float


class SoakMonitor:
    """Evaluate status snapshots without reading frames or commanding motion."""

    def __init__(self, config: SoakConfig) -> None:
        self.config = config
        self.started_at_s: float | None = None
        self.last_sample_at_s: float | None = None
        self.samples = 0
        self.healthy_samples = 0
        self.failure_counts: Counter[str] = Counter()
        self.generation_changes = 0
        self._generation: str | None = None
        self._progress: dict[str, _ProgressMarker] = {}
        self.recognition_samples = 0
        self.recognition_matches = 0
        self.recognition_confidences: list[float] = []
        self.metrics: dict[str, list[float]] = {
            "broker_frame_age_s": [],
            "runtime_frame_age_s": [],
            "detector_age_s": [],
            "broker_fps": [],
            "runtime_fps": [],
            "inference_ms": [],
            "main_status_request_ms": [],
            "camera_status_request_ms": [],
            "sample_gap_s": [],
        }

    def observe(
        self,
        *,
        now_s: float,
        main_status: dict[str, Any],
        camera_status: dict[str, Any],
        request_errors: list[str] | None = None,
        request_metrics: dict[str, float] | None = None,
    ) -> dict[str, object]:
        if self.started_at_s is None:
            self.started_at_s = now_s
        sample_gap_s = (
            None
            if self.last_sample_at_s is None
            else max(0.0, now_s - self.last_sample_at_s)
        )
        self.last_sample_at_s = now_s
        self.samples += 1
        errors = list(request_errors or ())
        timings = request_metrics or {}
        for name in ("main_status_request_ms", "camera_status_request_ms"):
            duration_ms = _number(timings.get(name))
            if duration_ms is not None:
                self._collect_metric(name, duration_ms)
                if duration_ms > self.config.maximum_status_request_s * 1000.0:
                    errors.append(f"{name}_slow")
        if sample_gap_s is not None:
            self._collect_metric("sample_gap_s", sample_gap_s)

        health = _mapping(main_status.get("health"))
        produce = _mapping(main_status.get("produce"))
        camera_rpc = _mapping(main_status.get("camera_rpc"))

        _require_true(camera_status, "ready", "broker_not_ready", errors)
        _require_equal(
            camera_status,
            "peer_state",
            "connected",
            "broker_peer_disconnected",
            errors,
        )
        _require_true(camera_status, "video_enabled", "broker_video_disabled", errors)
        _require_true(health, "camera_live", "runtime_camera_not_live", errors)
        _require_true(health, "produce_live", "detector_not_live", errors)
        _require_true(health, "gpu_ready", "detector_gpu_not_ready", errors)

        generation = _string(camera_status.get("generation"))
        runtime_generation = _string(main_status.get("camera_stream_generation"))
        rpc_generation = _string(camera_rpc.get("generation"))
        if generation is None:
            errors.append("broker_generation_missing")
        else:
            if self._generation is None:
                self._generation = generation
            elif generation != self._generation:
                self.generation_changes += 1
                self._generation = generation
                self._progress.clear()
            if runtime_generation != generation:
                errors.append("runtime_generation_mismatch")
            if rpc_generation != generation:
                errors.append("camera_rpc_generation_mismatch")
        if self.generation_changes > self.config.maximum_generation_changes:
            errors.append("too_many_generation_changes")

        self._check_age(
            "broker_frame_age_s",
            camera_status.get("last_frame_age_s"),
            self.config.maximum_camera_age_s,
            errors,
        )
        self._check_age(
            "runtime_frame_age_s",
            main_status.get("frame_age_s"),
            self.config.maximum_camera_age_s,
            errors,
        )
        self._check_age(
            "camera_source_age_s",
            camera_rpc.get("last_source_age_s"),
            self.config.maximum_camera_age_s,
            errors,
        )
        self._check_age(
            "detector_age_s",
            produce.get("age_s"),
            self.config.maximum_detection_age_s,
            errors,
        )

        self._check_progress(
            "broker_frame",
            camera_status.get("frame_id"),
            now_s,
            errors,
        )
        self._check_progress(
            "runtime_frame",
            main_status.get("frame_count"),
            now_s,
            errors,
        )
        self._check_progress(
            "detector_frame",
            produce.get("frame_id"),
            now_s,
            errors,
        )

        self._collect_metric("broker_fps", camera_status.get("fps_5s"))
        self._collect_metric("runtime_fps", main_status.get("camera_fps"))
        self._collect_metric("inference_ms", produce.get("inference_ms"))

        expected_match: dict[str, object] | None = None
        recognition_eligible = bool(
            self.config.expected_fruit
            and self.started_at_s is not None
            and now_s - self.started_at_s >= self.config.recognition_warmup_s
        )
        if recognition_eligible:
            self.recognition_samples += 1
            expected_match = self._best_expected_detection(produce)
            if expected_match is not None:
                self.recognition_matches += 1
                confidence = _number(expected_match.get("confidence"))
                if confidence is not None:
                    self.recognition_confidences.append(confidence)

        errors = sorted(set(errors))
        if not errors:
            self.healthy_samples += 1
        else:
            self.failure_counts.update(errors)
        return {
            "type": "sample",
            "elapsed_s": round(now_s - self.started_at_s, 3),
            "healthy": not errors,
            "errors": errors,
            "generation": generation,
            "broker_frame_id": camera_status.get("frame_id"),
            "runtime_frame_count": main_status.get("frame_count"),
            "detector_frame_id": produce.get("frame_id"),
            "broker_frame_age_s": camera_status.get("last_frame_age_s"),
            "runtime_frame_age_s": main_status.get("frame_age_s"),
            "detector_age_s": produce.get("age_s"),
            "broker_fps": camera_status.get("fps_5s"),
            "runtime_fps": main_status.get("camera_fps"),
            "inference_ms": produce.get("inference_ms"),
            "main_status_request_ms": timings.get("main_status_request_ms"),
            "camera_status_request_ms": timings.get("camera_status_request_ms"),
            "sample_gap_s": None if sample_gap_s is None else round(sample_gap_s, 4),
            "detections": produce.get("detections") or [],
            "expected_fruit": self.config.expected_fruit,
            "expected_match": expected_match,
        }

    def summary(self) -> dict[str, object]:
        elapsed = (
            0.0
            if self.started_at_s is None or self.last_sample_at_s is None
            else max(0.0, self.last_sample_at_s - self.started_at_s)
        )
        recognition_ratio = (
            None
            if self.recognition_samples == 0
            else self.recognition_matches / self.recognition_samples
        )
        recognition_passed = bool(
            self.config.expected_fruit is None
            or (
                recognition_ratio is not None
                and recognition_ratio >= self.config.minimum_recognition_ratio
            )
        )
        passed = bool(
            self.samples > 0
            and not self.failure_counts
            and self.generation_changes <= self.config.maximum_generation_changes
            and recognition_passed
        )
        return {
            "type": "summary",
            "passed": passed,
            "duration_s": round(elapsed, 3),
            "samples": self.samples,
            "healthy_samples": self.healthy_samples,
            "failed_samples": self.samples - self.healthy_samples,
            "failure_counts": dict(sorted(self.failure_counts.items())),
            "generation_changes": self.generation_changes,
            "expected_fruit": self.config.expected_fruit,
            "recognition_samples": self.recognition_samples,
            "recognition_matches": self.recognition_matches,
            "recognition_ratio": None
            if recognition_ratio is None
            else round(recognition_ratio, 4),
            "minimum_recognition_ratio": self.config.minimum_recognition_ratio,
            "recognition_confidence": _stats(self.recognition_confidences),
            "metrics": {name: _stats(values) for name, values in self.metrics.items()},
        }

    def _check_age(
        self,
        name: str,
        value: object,
        maximum_s: float,
        errors: list[str],
    ) -> None:
        age = _number(value)
        if age is None:
            errors.append(f"{name}_missing")
            return
        self._collect_metric(name, age)
        if age < 0.0 or age > maximum_s:
            errors.append(f"{name}_stale")

    def _check_progress(
        self,
        name: str,
        value: object,
        now_s: float,
        errors: list[str],
    ) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            errors.append(f"{name}_id_missing")
            return
        marker = self._progress.get(name)
        if marker is None or marker.value != value:
            self._progress[name] = _ProgressMarker(value, now_s)
            return
        if now_s - marker.changed_at_s > self.config.maximum_frame_stall_s:
            errors.append(f"{name}_frozen")

    def _collect_metric(self, name: str, value: object) -> None:
        number = _number(value)
        if number is not None:
            self.metrics.setdefault(name, []).append(number)

    def _best_expected_detection(
        self, produce: dict[str, Any]
    ) -> dict[str, object] | None:
        expected = self.config.expected_fruit
        if expected is None:
            return None
        thresholds = _mapping(produce.get("class_thresholds"))
        threshold = _number(thresholds.get(expected)) or 0.0
        matches: list[dict[str, object]] = []
        detections = produce.get("detections")
        if isinstance(detections, list):
            for item in detections:
                detection = _mapping(item)
                confidence = _number(detection.get("confidence"))
                if (
                    detection.get("label") == expected
                    and confidence is not None
                    and confidence >= threshold
                ):
                    matches.append(detection)
        return max(
            matches,
            key=lambda item: _number(item.get("confidence")) or 0.0,
            default=None,
        )


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    rendered = float(value)
    return rendered if math.isfinite(rendered) else None


def _require_true(
    payload: dict[str, Any], key: str, error: str, errors: list[str]
) -> None:
    if payload.get(key) is not True:
        errors.append(error)


def _require_equal(
    payload: dict[str, Any],
    key: str,
    expected: object,
    error: str,
    errors: list[str],
) -> None:
    if payload.get(key) != expected:
        errors.append(error)


def _stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "minimum": round(min(values), 4),
        "mean": round(statistics.fmean(values), 4),
        "maximum": round(max(values), 4),
    }


def fetch_json(url: str, timeout_s: float) -> dict[str, Any]:
    request = urlrequest.Request(url, headers={"Accept": "application/json"})
    try:
        with urlrequest.urlopen(request, timeout=timeout_s) as response:
            payload = json.load(response)
    except (OSError, ValueError, urlerror.URLError) as exc:
        raise RuntimeError(f"{url}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"{url}: response is not a JSON object")
    return payload


def parse_args() -> argparse.Namespace:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(
        description=(
            "Poll Woof camera and detector telemetry without commanding motion"
        )
    )
    parser.add_argument("--main-url", default="http://woof.local:8096")
    parser.add_argument("--voice-url", default="http://woof.local:8098")
    parser.add_argument("--duration-s", type=float, default=1800.0)
    parser.add_argument("--interval-s", type=float, default=0.5)
    parser.add_argument("--request-timeout-s", type=float, default=1.0)
    parser.add_argument("--maximum-camera-age-s", type=float, default=0.75)
    parser.add_argument("--maximum-detection-age-s", type=float, default=0.75)
    parser.add_argument("--maximum-frame-stall-s", type=float, default=1.5)
    parser.add_argument("--maximum-status-request-s", type=float, default=1.0)
    parser.add_argument("--maximum-generation-changes", type=int, default=0)
    parser.add_argument("--expected-fruit", choices=TARGETS)
    parser.add_argument("--minimum-recognition-ratio", type=float, default=0.80)
    parser.add_argument("--recognition-warmup-s", type=float, default=3.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/soak") / f"camera-fruit-{timestamp}.jsonl",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.duration_s <= 0.0 or args.interval_s <= 0.0:
        raise ValueError("duration and interval must be positive")
    if args.request_timeout_s <= 0.0:
        raise ValueError("request timeout must be positive")
    config = SoakConfig(
        maximum_camera_age_s=args.maximum_camera_age_s,
        maximum_detection_age_s=args.maximum_detection_age_s,
        maximum_frame_stall_s=args.maximum_frame_stall_s,
        maximum_status_request_s=args.maximum_status_request_s,
        maximum_generation_changes=args.maximum_generation_changes,
        expected_fruit=args.expected_fruit,
        minimum_recognition_ratio=args.minimum_recognition_ratio,
        recognition_warmup_s=args.recognition_warmup_s,
    )
    monitor = SoakMonitor(config)
    main_status_url = f"{args.main_url.rstrip('/')}/api/status"
    camera_status_url = f"{args.voice_url.rstrip('/')}/api/camera/status"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + args.duration_s
    next_sample = started
    last_progress_at = -math.inf

    with args.output.open("w", encoding="utf-8") as report:
        metadata = {
            "type": "metadata",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "main_status_url": main_status_url,
            "camera_status_url": camera_status_url,
            "duration_s": args.duration_s,
            "interval_s": args.interval_s,
            "config": asdict(config),
        }
        report.write(json.dumps(metadata, sort_keys=True) + "\n")
        try:
            while time.monotonic() < deadline:
                request_errors: list[str] = []
                request_metrics: dict[str, float] = {}
                request_started = time.perf_counter()
                try:
                    main_status = fetch_json(main_status_url, args.request_timeout_s)
                except (RuntimeError, TypeError) as exc:
                    main_status = {}
                    request_errors.append(f"main_status_request_failed:{exc}")
                request_metrics["main_status_request_ms"] = round(
                    (time.perf_counter() - request_started) * 1000.0,
                    3,
                )
                request_started = time.perf_counter()
                try:
                    camera_status = fetch_json(
                        camera_status_url, args.request_timeout_s
                    )
                except (RuntimeError, TypeError) as exc:
                    camera_status = {}
                    request_errors.append(f"camera_status_request_failed:{exc}")
                request_metrics["camera_status_request_ms"] = round(
                    (time.perf_counter() - request_started) * 1000.0,
                    3,
                )
                now = time.monotonic()
                sample = monitor.observe(
                    now_s=now,
                    main_status=main_status,
                    camera_status=camera_status,
                    request_errors=request_errors,
                    request_metrics=request_metrics,
                )
                report.write(json.dumps(sample, sort_keys=True) + "\n")
                report.flush()
                elapsed = now - started
                if sample["errors"] or elapsed - last_progress_at >= 5.0:
                    state = "PASS" if sample["healthy"] else "FAIL"
                    print(
                        f"{state} elapsed={elapsed:.1f}s "
                        f"generation={sample['generation']} "
                        f"camera_age={sample['broker_frame_age_s']} "
                        f"detector_age={sample['detector_age_s']} "
                        f"main_ms={sample['main_status_request_ms']} "
                        f"camera_ms={sample['camera_status_request_ms']} "
                        f"errors={','.join(sample['errors']) or '-'}",
                        flush=True,
                    )
                    last_progress_at = elapsed
                next_sample += args.interval_s
                time.sleep(max(0.0, next_sample - time.monotonic()))
        except KeyboardInterrupt:
            print("Probe interrupted; writing partial summary", file=sys.stderr)
        summary = monitor.summary()
        summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        summary["output"] = str(args.output)
        report.write(json.dumps(summary, sort_keys=True) + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    try:
        summary = run(parse_args())
    except (RuntimeError, TypeError, ValueError) as exc:
        print(f"camera-fruit soak configuration failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
