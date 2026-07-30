from __future__ import annotations

import math

import pytest

from collie_demo.temporal_fusion import (
    MaxMojoTemporalFusion,
    ReferenceTemporalFusion,
    TemporalFusionRequest,
    compare_temporal_results,
)


def request(
    *,
    previous_x: float = 100.0,
    velocity_x: float = 0.0,
    measurement_x: float = 100.0,
    measurement_valid: bool = True,
    confidence: float = 0.8,
    delta_s: float = 0.1,
) -> TemporalFusionRequest:
    return TemporalFusionRequest(
        previous_bbox_xywh=(previous_x, 80.0, 60.0, 50.0),
        previous_velocity_xywh_s=(velocity_x, 0.0, 0.0, 0.0),
        measurement_bbox_xywh=(measurement_x, 80.0, 60.0, 50.0),
        frame_width=640,
        frame_height=480,
        delta_s=delta_s,
        confidence=confidence,
        measurement_valid=measurement_valid,
    )


def test_reference_smooths_alternating_detector_jitter() -> None:
    backend = ReferenceTemporalFusion()
    bbox = (100.0, 80.0, 60.0, 50.0)
    velocity = (0.0, 0.0, 0.0, 0.0)
    raw_positions = [112.0, 100.0, 112.0, 100.0, 112.0, 100.0]
    fused_positions: list[float] = []

    for raw_x in raw_positions:
        result = backend.process(
            [
                TemporalFusionRequest(
                    previous_bbox_xywh=bbox,
                    previous_velocity_xywh_s=velocity,
                    measurement_bbox_xywh=(raw_x, 80.0, 60.0, 50.0),
                    frame_width=640,
                    frame_height=480,
                    delta_s=0.1,
                    confidence=0.7,
                    measurement_valid=True,
                )
            ]
        )[0]
        assert result.valid
        assert result.measurement_used
        bbox = result.bbox_xywh
        velocity = result.velocity_xywh_s
        fused_positions.append(bbox[0])

    raw_steps = [
        abs(current - previous)
        for previous, current in zip(raw_positions, raw_positions[1:])
    ]
    fused_steps = [
        abs(current - previous)
        for previous, current in zip(fused_positions, fused_positions[1:])
    ]
    raw_rms = math.sqrt(sum(step * step for step in raw_steps) / len(raw_steps))
    fused_rms = math.sqrt(
        sum(step * step for step in fused_steps) / len(fused_steps)
    )
    assert fused_rms < raw_rms


def test_reference_bridges_a_short_detector_gap_with_decayed_velocity() -> None:
    result = ReferenceTemporalFusion().process(
        [
            request(
                previous_x=100.0,
                velocity_x=30.0,
                measurement_x=0.0,
                measurement_valid=False,
                delta_s=0.1,
            )
        ]
    )[0]

    assert result.valid
    assert not result.measurement_used
    assert result.bbox_xywh[0] == pytest.approx(103.0)
    assert result.velocity_xywh_s[0] == pytest.approx(23.4)


def test_reference_rejects_implausible_measurement_jump() -> None:
    result = ReferenceTemporalFusion().process(
        [request(previous_x=20.0, measurement_x=600.0)]
    )[0]

    assert result.valid
    assert not result.measurement_used
    assert result.bbox_xywh[0] == pytest.approx(20.0)


def test_max_mojo_matches_reference_for_measurement_and_gap() -> None:
    pytest.importorskip("max")
    requests = [
        request(previous_x=100.0, velocity_x=8.0, measurement_x=106.0),
        request(
            previous_x=220.0,
            velocity_x=-12.0,
            measurement_x=0.0,
            measurement_valid=False,
        ),
    ]
    reference = ReferenceTemporalFusion().process(requests)
    backend = MaxMojoTemporalFusion(device="cpu")

    candidate = backend.process(requests)
    matches, maximum_error = compare_temporal_results(reference, candidate)

    assert matches
    assert maximum_error <= 1e-3
    assert backend.status()["executions"] == 1
