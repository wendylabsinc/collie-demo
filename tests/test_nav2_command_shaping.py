from collie_nav2.command_shaping import apply_measured_motion_floors


def shape(forward_mps: float, yaw_rps: float = 0.0) -> tuple[float, float]:
    return apply_measured_motion_floors(
        forward_mps,
        yaw_rps,
        minimum_forward_mps=0.25,
        minimum_rotation_yaw_rps=0.35,
    )


def test_positive_forward_commands_use_measured_working_floor() -> None:
    assert shape(0.05) == (0.25, 0.0)
    assert shape(0.249) == (0.25, 0.0)
    assert shape(0.25) == (0.25, 0.0)
    assert shape(0.30) == (0.30, 0.0)
    assert shape(0.50) == (0.50, 0.0)


def test_stop_reverse_and_rotation_only_are_not_promoted_forward() -> None:
    assert shape(0.0) == (0.0, 0.0)
    assert shape(-0.05) == (-0.05, 0.0)
    assert shape(0.01, 0.10) == (0.01, 0.35)
    assert shape(0.01, -0.10) == (0.01, -0.35)


def test_translation_with_steering_keeps_yaw_and_raises_forward() -> None:
    assert shape(0.05, 0.10) == (0.25, 0.10)
