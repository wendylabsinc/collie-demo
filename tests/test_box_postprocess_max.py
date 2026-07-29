from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("max")

from collie_demo.box_postprocess import (
    BoxPostprocessRequest,
    MaxMojoBoxPostprocessor,
    ReferenceBoxPostprocessor,
    compare_box_postprocess_results,
)


def test_max_mojo_cpu_matches_reference() -> None:
    request = BoxPostprocessRequest.from_affine(
        bbox_xywh=(70.0, 80.0, 90.0, 70.0),
        affine_2x3=np.asarray(
            [[1.01, -0.02, 9.0], [0.02, 1.01, 6.0]],
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
    reference = ReferenceBoxPostprocessor().process(request)
    backend = MaxMojoBoxPostprocessor(device="cpu")

    actual = backend.process(request)
    comparison = compare_box_postprocess_results(reference, actual)

    assert comparison.matches
    assert comparison.maximum_absolute_error <= 1e-3
    assert backend.status()["executions"] == 1
