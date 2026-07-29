from __future__ import annotations

import argparse
import json
import math
import time
from statistics import median

import numpy as np

from .box_postprocess import (
    BoxPostprocessRequest,
    MaxMojoBoxPostprocessor,
    ReferenceBoxPostprocessor,
    compare_box_postprocess_results,
)


def _request(index: int) -> BoxPostprocessRequest:
    angle = math.sin(index * 0.17) * 0.035
    scale = 1.0 + math.sin(index * 0.11) * 0.025
    cosine = math.cos(angle) * scale
    sine = math.sin(angle) * scale
    transform = np.asarray(
        [
            [cosine, -sine, math.sin(index * 0.07) * 4.0],
            [sine, cosine, math.cos(index * 0.09) * 2.0],
        ],
        dtype=np.float32,
    )
    return BoxPostprocessRequest.from_affine(
        bbox_xywh=(540.0, 450.0, 120.0, 90.0),
        affine_2x3=transform,
        frame_width=1280,
        frame_height=720,
        minimum_box_size_px=6.0,
        minimum_scale_per_frame=0.70,
        maximum_scale_per_frame=1.40,
        maximum_center_step_fraction=0.35,
        new_box_weight=1.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Shadow-compare MAX/Mojo box postprocessing"
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "accelerator"),
        default="cpu",
    )
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()
    if args.iterations < 1:
        raise SystemExit("--iterations must be positive")

    reference = ReferenceBoxPostprocessor()
    candidate = MaxMojoBoxPostprocessor(device=args.device)
    timings_ms: list[float] = []
    mismatches = 0
    maximum_error = 0.0
    for index in range(args.iterations):
        request = _request(index)
        expected = reference.process(request)
        started = time.perf_counter()
        actual = candidate.process(request)
        timings_ms.append((time.perf_counter() - started) * 1000.0)
        comparison = compare_box_postprocess_results(expected, actual)
        maximum_error = max(maximum_error, comparison.maximum_absolute_error)
        if not comparison.matches:
            mismatches += 1

    ordered = sorted(timings_ms)
    p95_index = int(0.95 * (len(ordered) - 1))
    report = {
        "iterations": args.iterations,
        "mismatches": mismatches,
        "maximum_absolute_error": round(maximum_error, 8),
        "p50_ms": round(median(timings_ms), 4),
        "p95_ms": round(ordered[p95_index], 4),
        "maximum_ms": round(max(timings_ms), 4),
        "backend": candidate.status(),
        "control_authority": "none_shadow_only",
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
