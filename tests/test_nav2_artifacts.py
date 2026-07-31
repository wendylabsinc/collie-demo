from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).parents[1]
NAV2 = ROOT / "nav2" / "autonomy"


def test_nav2_tree_is_forward_only_and_has_bounded_recovery() -> None:
    tree = (
        NAV2
        / "ros2_ws"
        / "src"
        / "collie_nav2"
        / "config"
        / "navigate_home.xml"
    ).read_text()

    ET.fromstring(tree)
    assert "<BackUp" not in tree
    assert "<Spin " in tree
    assert "<Wait " in tree
    assert "ClearEntireCostmap" in tree


def test_nav2_internal_goal_margin_is_stricter_than_external_contract() -> None:
    params = (
        NAV2
        / "ros2_ws"
        / "src"
        / "collie_nav2"
        / "config"
        / "nav2_params.yaml"
    ).read_text()

    assert "default_server_timeout: 2000" in params
    assert "movement_time_allowance: 12.0" in params
    assert "xy_goal_tolerance: 0.08" in params
    assert "yaw_goal_tolerance: 0.0698132" in params
    assert "RegulatedPurePursuitController" in params
    assert "desired_linear_vel: 0.25" in params
    assert "allow_reversing: false" in params
    assert "use_collision_detection: true" in params
    assert "enable_stamped_cmd_vel: false" in params


def test_nav2_gateway_requires_active_lifecycle_and_tracks_heading_progress() -> None:
    gateway = (
        NAV2
        / "ros2_ws"
        / "src"
        / "collie_nav2"
        / "collie_nav2"
        / "gateway.py"
    ).read_text()

    assert '"/bt_navigator/get_state"' in gateway
    assert "State.PRIMARY_STATE_ACTIVE" in gateway
    assert "STALL_HEADING_PROGRESS_RAD" in gateway
    assert "_last_linear_progress_pose" in gateway
    assert "Only map-frame pose" in gateway
    assert "Nav2 command exceeded the guarded stage envelope" in gateway


def test_rtabmap_uses_filtered_hesai_scan_instead_of_raw_self_returns() -> None:
    launch = (
        NAV2
        / "ros2_ws"
        / "src"
        / "collie_nav2"
        / "launch"
        / "bringup.launch.py"
    ).read_text()

    assert '"subscribe_scan": True' in launch
    assert '"subscribe_scan_cloud": False' in launch
    assert '("scan", "/scan")' in launch
    assert '("scan_cloud", "/pointcloud")' not in launch


def test_normal_bringup_uses_only_the_rtabmap_localization_path() -> None:
    launch = (
        NAV2
        / "ros2_ws"
        / "src"
        / "collie_nav2"
        / "launch"
        / "bringup.launch.py"
    ).read_text()

    assert 'executable="rtabmap"' in launch
    assert '"map_frame_id": "map"' in launch
    assert '("odom", "/odom")' in launch
    assert "collie_hesai_icp_odometry" not in launch
    assert "hesai_odom_shadow" not in launch
    assert "collie_amcl_shadow" not in launch
    assert "amcl_shadow_pose" not in launch


def test_stage_image_defaults_to_map_frame_nav2_return() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text()

    assert "COLLIE_RETURN_HOME_ENABLED=1" in dockerfile
    assert "COLLIE_RETURN_BACKEND=nav2" in dockerfile
    assert "COLLIE_RETURN_ARRIVAL_TOLERANCE_M=0.10" in dockerfile
    assert "COLLIE_RETURN_HEADING_TOLERANCE_DEG=5.0" in dockerfile


def test_sensor_bridge_prioritizes_hesai_over_high_rate_state_topics() -> None:
    bridge = (ROOT / "nav2" / "bridge" / "bridge.py").read_text()
    manifest = (ROOT / "nav2" / "wendy.json").read_text()

    assert "SingleThreadedExecutor" in bridge
    assert "MultiThreadedExecutor" not in bridge
    assert bridge.count("depth=1") >= 2
    assert "LaserScan" in bridge
    assert "PointCloud2" not in bridge
    assert "Imu" not in bridge
    assert "RAW_SCAN_TOPIC" in bridge
    assert "ODOM_OUTPUT_MAX_HZ" in bridge
    assert '"ODOM_OUTPUT_MAX_HZ": "50"' in manifest
    assert "create_publisher" in bridge
    assert "cmd_vel" not in bridge


def test_hesai_cloud_is_filtered_before_the_cross_domain_bridge() -> None:
    entrypoint = (ROOT / "nav2" / "bridge" / "entrypoint.sh").read_text()
    launch = (
        NAV2
        / "ros2_ws"
        / "src"
        / "collie_nav2"
        / "launch"
        / "bringup.launch.py"
    ).read_text()

    assert "pointcloud_to_laserscan pointcloud_to_laserscan_node" in entrypoint
    assert "target_frame:=base_link" in entrypoint
    assert "min_height:=0.05" in entrypoint
    assert "max_height:=0.90" in entrypoint
    assert "collie_pointcloud_to_scan" not in launch


def test_hesai_mount_uses_the_read_only_validated_transform() -> None:
    manifest = (ROOT / "nav2" / "wendy.json").read_text()

    assert '"LIDAR_SOURCE_TOPIC": "/hesai/points"' in manifest
    assert '"COLLIE_LIDAR_TRANSFORM_VALIDATED": "1"' in manifest


def test_hesai_calibration_is_read_only_and_never_unlocks_motion() -> None:
    capture = (
        ROOT
        / "nav2"
        / "bridge"
        / "tools"
        / "capture_extrinsic_pair.py"
    ).read_text()
    estimator = (
        ROOT / "nav2" / "tools" / "estimate_hesai_extrinsic.py"
    ).read_text()

    assert "create_subscription" in capture
    assert "create_publisher" not in capture
    assert "motion_unlock_authorized" in estimator
    assert '"motion_unlock_authorized": False' in estimator


def test_live_map_probe_is_read_only() -> None:
    probe = (
        NAV2 / "integration" / "live_map_probe.py"
    ).read_text()

    assert "create_subscription" in probe
    assert "create_publisher" not in probe
    assert '"read_only": True' in probe


def test_gateway_brakes_during_bounded_nav2_recovery_gaps() -> None:
    gateway = (
        NAV2
        / "ros2_ws"
        / "src"
        / "collie_nav2"
        / "collie_nav2"
        / "gateway.py"
    ).read_text()
    assert "COMMAND_TRANSITION_GRACE_S = 3.0" in gateway
    assert "command_age > self.COMMAND_MAX_AGE_S" in gateway
    assert "forward = 0.0" in gateway
    assert "yaw = 0.0" in gateway


def test_gateway_applies_a_minimum_only_to_fresh_rotation_commands() -> None:
    gateway = (
        NAV2
        / "ros2_ws"
        / "src"
        / "collie_nav2"
        / "collie_nav2"
        / "gateway.py"
    ).read_text()
    assert "MIN_ROTATION_YAW_RPS = 0.35" in gateway
    assert "abs(forward) <= 0.02" in gateway
    assert "math.copysign(self.MIN_ROTATION_YAW_RPS, yaw)" in gateway


def test_synthetic_trial_can_disable_only_the_mapping_process() -> None:
    launch = (
        NAV2
        / "ros2_ws"
        / "src"
        / "collie_nav2"
        / "launch"
        / "bringup.launch.py"
    ).read_text()
    dockerfile = (NAV2 / "Dockerfile").read_text()
    assert 'os.environ.get("COLLIE_NAV2_MAPPING_ENABLED", "1")' in launch
    assert "if mapping_enabled:" in launch
    assert "COLLIE_NAV2_MAPPING_ENABLED=1" in dockerfile


def test_synthetic_trial_covers_success_and_required_failure_modes() -> None:
    trial = (NAV2 / "integration" / "synthetic_return_trial.py").read_text()

    for scenario in (
        "success",
        "detour",
        "localization-loss",
        "no-path",
        "stalled-motion",
    ):
        assert f'"{scenario}"' in trial
    assert "motion owner did not confirm STOP" in trial
