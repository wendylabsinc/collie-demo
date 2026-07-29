from __future__ import annotations

import math
import os
from pathlib import Path

import uvicorn

from .app import create_app
from .camera import create_camera
from .controller import ApproachConfig, ApproachController
from .fruit import FruitDetector
from .heading import SportModeHeadingProvider
from .mission import MissionConfig
from .motion import MotionConfig, create_motion, initialize_dds
from .pointing import PointingPolicyConfig, PointingPolicyManager
from .runtime import CollieRuntime


def env_bool(name: str, default: bool = False) -> bool:
    return os.environ.get(name, "1" if default else "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def env_class_thresholds(name: str) -> dict[str, float]:
    rendered = os.environ.get(name, "").strip()
    if not rendered:
        return {}
    thresholds: dict[str, float] = {}
    for item in rendered.split(","):
        label, separator, value = item.partition("=")
        if not separator or not label.strip() or not value.strip():
            raise ValueError(f"{name} must use label=value pairs")
        thresholds[label.strip()] = float(value)
    return thresholds


def build_runtime() -> CollieRuntime:
    network_interface = os.environ.get("GO2_NETWORK_INTERFACE", "").strip() or None
    motion_enabled = env_bool("COLLIE_MOTION_ENABLED")
    allow_unranged = env_bool("COLLIE_ALLOW_UNRANGED_DEMO")
    produce_model = Path(
        os.environ.get(
            "COLLIE_PRODUCE_MODEL",
            "models/snapstock/fruit_vegetable_yolov8m.pt",
        )
    )
    initialize_dds(network_interface)
    mission_config = MissionConfig(
        enabled=env_bool("COLLIE_MEMORY_DEMO_ENABLED"),
        autonomous_turn_enabled=env_bool("COLLIE_AUTONOMOUS_TURN_ENABLED"),
        direct_turn_enabled=env_bool("COLLIE_DIRECT_TURN_ENABLED"),
        initial_hello_enabled=env_bool("COLLIE_INITIAL_HELLO_ENABLED"),
        match_stretch_enabled=env_bool("COLLIE_MATCH_STRETCH_ENABLED"),
        match_stretch_settle_s=float(
            os.environ.get("COLLIE_MATCH_STRETCH_SETTLE_S", "3.5")
        ),
        match_reacquire_timeout_s=float(
            os.environ.get("COLLIE_MATCH_REACQUIRE_TIMEOUT_S", "3.0")
        ),
        arrival_hello_enabled=env_bool("COLLIE_ARRIVAL_HELLO_ENABLED"),
        arrival_hello_settle_s=float(
            os.environ.get("COLLIE_ARRIVAL_HELLO_SETTLE_S", "0.35")
        ),
        arrival_rest_enabled=env_bool("COLLIE_ARRIVAL_REST_ENABLED"),
        arrival_rest_duration_s=float(
            os.environ.get("COLLIE_ARRIVAL_REST_DURATION_S", "5.0")
        ),
        arrival_pointing_enabled=env_bool("COLLIE_ARRIVAL_POINTING_ENABLED"),
        arrival_pointing_label=os.environ.get(
            "COLLIE_ARRIVAL_POINTING_LABEL", "pear"
        ),
        arrival_pointing_timeout_s=float(
            os.environ.get("COLLIE_ARRIVAL_POINTING_TIMEOUT_S", "20.0")
        ),
        return_home_enabled=env_bool("COLLIE_RETURN_HOME_ENABLED"),
        return_arrival_tolerance_m=float(
            os.environ.get("COLLIE_RETURN_ARRIVAL_TOLERANCE_M", "0.25")
        ),
        return_heading_tolerance_rad=math.radians(
            float(os.environ.get("COLLIE_RETURN_HEADING_TOLERANCE_DEG", "10"))
        ),
        return_heading_gate_rad=math.radians(
            float(os.environ.get("COLLIE_RETURN_HEADING_GATE_DEG", "30"))
        ),
        return_forward_mps=float(
            os.environ.get("COLLIE_RETURN_FORWARD_MPS", "0.20")
        ),
        return_yaw_gain=float(
            os.environ.get("COLLIE_RETURN_YAW_GAIN", "1.2")
        ),
        return_timeout_s=float(
            os.environ.get("COLLIE_RETURN_TIMEOUT_S", "20")
        ),
        return_stall_timeout_s=float(
            os.environ.get("COLLIE_RETURN_STALL_TIMEOUT_S", "3.0")
        ),
        return_stall_min_progress_m=float(
            os.environ.get("COLLIE_RETURN_STALL_MIN_PROGRESS_M", "0.06")
        ),
        capture_timeout_s=float(
            os.environ.get("COLLIE_MEMORY_CAPTURE_TIMEOUT_S", "2.0")
        ),
        match_confirmations_required=int(
            os.environ.get("COLLIE_MATCH_CONFIRMATIONS", "3")
        ),
        approach_misses_allowed=int(
            os.environ.get("COLLIE_APPROACH_MATCH_MISSES", "2")
        ),
        turn_angle_rad=math.radians(
            float(os.environ.get("COLLIE_TURN_ANGLE_DEG", "180"))
        ),
        turn_rate_rps=float(os.environ.get("COLLIE_TURN_RATE_RPS", "0.30")),
        turn_tolerance_rad=math.radians(
            float(os.environ.get("COLLIE_TURN_TOLERANCE_DEG", "7"))
        ),
        turn_timeout_s=float(os.environ.get("COLLIE_TURN_TIMEOUT_S", "15")),
        turn_stall_timeout_s=float(
            os.environ.get("COLLIE_TURN_STALL_TIMEOUT_S", "2.0")
        ),
        turn_stall_min_progress_rad=math.radians(
            float(os.environ.get("COLLIE_TURN_STALL_MIN_PROGRESS_DEG", "10"))
        ),
        search_rate_rps=float(os.environ.get("COLLIE_SEARCH_RATE_RPS", "0.20")),
        search_sweep_rad=math.radians(
            float(os.environ.get("COLLIE_SEARCH_SWEEP_DEG", "75"))
        ),
        search_timeout_s=float(
            os.environ.get("COLLIE_SEARCH_TIMEOUT_S", "9")
        ),
        near_bottom_ratio=float(
            os.environ.get("COLLIE_NEAR_BOTTOM_RATIO", "0.86")
        ),
        near_center_ratio=float(
            os.environ.get("COLLIE_NEAR_CENTER_RATIO", "0.72")
        ),
        near_bbox_height_ratio=float(
            os.environ.get("COLLIE_NEAR_BBOX_HEIGHT_RATIO", "0.15")
        ),
        near_confirmations_required=int(
            os.environ.get("COLLIE_NEAR_CONFIRMATIONS", "3")
        ),
        near_loss_grace_s=float(
            os.environ.get("COLLIE_NEAR_LOSS_GRACE_S", "0.75")
        ),
        final_approach_distance_m=float(
            os.environ.get("COLLIE_FINAL_APPROACH_DISTANCE_M", "0.10")
        ),
        final_approach_mps=float(
            os.environ.get("COLLIE_FINAL_APPROACH_MPS", "0.10")
        ),
        final_approach_timeout_s=float(
            os.environ.get("COLLIE_FINAL_APPROACH_TIMEOUT_S", "3.0")
        ),
        final_approach_stall_timeout_s=float(
            os.environ.get("COLLIE_FINAL_APPROACH_STALL_TIMEOUT_S", "1.0")
        ),
        final_approach_stall_min_progress_m=float(
            os.environ.get("COLLIE_FINAL_APPROACH_STALL_MIN_PROGRESS_M", "0.01")
        ),
    )
    controller_config = ApproachConfig(
        stable_frames_required=int(os.environ.get("COLLIE_STABLE_FRAMES", "3")),
        maximum_target_age_s=float(
            os.environ.get("COLLIE_MAX_TARGET_AGE_S", "0.35")
        ),
        forward_mps=float(os.environ.get("COLLIE_FORWARD_MPS", "0.08")),
        forward_budget_s=float(os.environ.get("COLLIE_FORWARD_BUDGET_S", "1.5")),
    )
    motion = (
        create_motion(
            MotionConfig(
                maximum_forward_mps=max(
                    controller_config.forward_mps,
                    mission_config.return_forward_mps,
                    mission_config.final_approach_mps,
                ),
                maximum_yaw_rps=max(
                    controller_config.maximum_yaw_rps,
                    mission_config.turn_rate_rps,
                    mission_config.search_rate_rps,
                    mission_config.return_yaw_gain
                    * mission_config.return_heading_gate_rad,
                ),
                remote_api_settle_s=float(
                    os.environ.get("COLLIE_REMOTE_API_SETTLE_S", "0.5")
                ),
                skill_timeout_s=float(
                    os.environ.get("COLLIE_SKILL_TIMEOUT_S", "12.0")
                ),
                client_timeout_s=float(
                    os.environ.get("COLLIE_CLIENT_TIMEOUT_S", "12.0")
                ),
            )
        )
        if motion_enabled
        else None
    )
    pointing = PointingPolicyManager(
        PointingPolicyConfig(
            enabled=env_bool("COLLIE_POINTING_ENABLED"),
            policy_path=Path(
                os.environ.get(
                    "COLLIE_POINTING_POLICY",
                    "models/pointing/policy_actor_42500.jit",
                )
            ),
            network_interface=network_interface or "enP8p1s0",
            status_url=os.environ.get(
                "COLLIE_POINTING_STATUS_URL",
                "http://127.0.0.1:8096/api/status",
            ),
            duration_s=float(
                os.environ.get("COLLIE_POINTING_DURATION_S", "1.0")
            ),
            action_gain=float(
                os.environ.get("COLLIE_POINTING_ACTION_GAIN", "1.0")
            ),
            maximum_target_rate_rad_s=float(
                os.environ.get("COLLIE_POINTING_MAX_RATE_RAD_S", "0.60")
            ),
            kp=float(os.environ.get("COLLIE_POINTING_KP", "25.0")),
            kd=float(os.environ.get("COLLIE_POINTING_KD", "0.5")),
        )
    )
    return CollieRuntime(
        camera=create_camera(),
        controller=ApproachController(controller_config),
        motion=motion,
        motion_enabled=motion_enabled,
        pointing=pointing,
        allow_unranged_forward=allow_unranged,
        produce_detector=FruitDetector(
            produce_model,
            confidence=float(os.environ.get("COLLIE_PRODUCE_CONFIDENCE", "0.5")),
            class_thresholds=env_class_thresholds(
                "COLLIE_PRODUCE_CLASS_THRESHOLDS"
            ),
            device=os.environ.get("COLLIE_INFERENCE_DEVICE", "").strip() or None,
            task=os.environ.get("COLLIE_PRODUCE_TASK", "").strip() or None,
        ),
        loop_hz=float(os.environ.get("COLLIE_CAMERA_HZ", "30")),
        annotated_hz=float(os.environ.get("COLLIE_ANNOTATED_HZ", "5")),
        produce_revalidation_misses_required=int(
            os.environ.get("COLLIE_REVALIDATION_MISSES", "3")
        ),
        maximum_produce_age_s=float(
            os.environ.get("COLLIE_MAX_PRODUCE_AGE_S", "0.75")
        ),
        follow_period_s=float(os.environ.get("COLLIE_FOLLOW_PERIOD_S", "0.05")),
        follow_start_timeout_s=float(
            os.environ.get("COLLIE_FOLLOW_START_TIMEOUT_S", "1.5")
        ),
        heading_provider=(
            SportModeHeadingProvider(
                maximum_age_s=float(
                    os.environ.get("COLLIE_HEADING_MAX_AGE_S", "0.5")
                )
            )
            if mission_config.enabled
            else None
        ),
        mission_config=mission_config,
    )


def main() -> None:
    web_directory = Path(
        os.environ.get("COLLIE_WEB_DIRECTORY", str(Path.cwd() / "web"))
    ).resolve()
    uvicorn.run(
        create_app(build_runtime(), web_directory),
        host=os.environ.get("COLLIE_BIND", "0.0.0.0"),
        port=int(os.environ.get("COLLIE_PORT", "8096")),
        log_level="info",
    )


if __name__ == "__main__":
    main()
