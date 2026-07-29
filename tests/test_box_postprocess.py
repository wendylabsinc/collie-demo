from __future__ import annotations

import numpy as np
import pytest

from collie_demo.box_postprocess import (
    BoxPostprocessRequest,
    BoxPostprocessResult,
    ReferenceBoxPostprocessor,
    compare_box_postprocess_results,
    shadow_box_postprocessor_from_mode,
)


def _request(
    transform: np.ndarray | None = None,
) -> BoxPostprocessRequest:
    if transform is None:
        transform = np.asarray(
            [[1.0, 0.0, 9.0], [0.0, 1.0, 6.0]],
            dtype=np.float32,
        )
    return BoxPostprocessRequest.from_affine(
        bbox_xywh=(70.0, 80.0, 90.0, 70.0),
        affine_2x3=transform,
        frame_width=320,
        frame_height=240,
        minimum_box_size_px=6.0,
        minimum_scale_per_frame=0.70,
        maximum_scale_per_frame=1.40,
        maximum_center_step_fraction=0.35,
        new_box_weight=1.0,
    )


def test_reference_box_postprocessor_translates_box() -> None:
    result = ReferenceBoxPostprocessor().process(_request())

    assert result.valid is True
    assert result.bbox_xywh == pytest.approx((79.0, 86.0, 90.0, 70.0))
    assert result.scale == pytest.approx(1.0)
    assert result.center_step_px == pytest.approx((9.0**2 + 6.0**2) ** 0.5)


def test_reference_box_postprocessor_clips_and_rejects_box() -> None:
    request = BoxPostprocessRequest.from_affine(
        bbox_xywh=(300.0, 220.0, 15.0, 15.0),
        affine_2x3=np.asarray(
            [[1.0, 0.0, 30.0], [0.0, 1.0, 30.0]],
            dtype=np.float32,
        ),
        frame_width=320,
        frame_height=240,
        minimum_box_size_px=6.0,
        minimum_scale_per_frame=0.70,
        maximum_scale_per_frame=1.40,
        maximum_center_step_fraction=0.35,
        new_box_weight=1.0,
    )

    result = ReferenceBoxPostprocessor().process(request)

    assert result.valid is False
    assert result.failure_reason == "box_left_frame"
    assert result.bbox_xywh[2:] == pytest.approx((0.0, 0.0))


def test_reference_box_postprocessor_rejects_scale() -> None:
    result = ReferenceBoxPostprocessor().process(
        _request(
            np.asarray(
                [[1.8, 0.0, 0.0], [0.0, 1.8, 0.0]],
                dtype=np.float32,
            )
        )
    )

    assert result.valid is False
    assert result.failure_reason == "implausible_scale"


def test_box_postprocess_comparison_checks_validity_and_values() -> None:
    reference = BoxPostprocessResult((1.0, 2.0, 3.0, 4.0), True, 1.0, 2.0)
    close = BoxPostprocessResult((1.0001, 2.0, 3.0, 4.0), True, 1.0, 2.0)
    wrong = BoxPostprocessResult((1.0, 2.0, 3.0, 4.0), False, 1.0, 2.0)

    assert compare_box_postprocess_results(reference, close).matches
    assert not compare_box_postprocess_results(reference, wrong).matches


def test_shadow_mode_is_explicit() -> None:
    assert shadow_box_postprocessor_from_mode("off") is None
    with pytest.raises(ValueError, match="must be off or max_mojo"):
        shadow_box_postprocessor_from_mode("replace_control")
