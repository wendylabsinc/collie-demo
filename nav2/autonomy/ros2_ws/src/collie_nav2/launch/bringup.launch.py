from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    share = get_package_share_directory("collie_nav2")
    nav2_params = os.path.join(share, "config", "nav2_params.yaml")
    navigation_tree = os.path.join(
        share, "config", "navigate_home.xml"
    )
    lidar_frame = os.environ.get("COLLIE_LIDAR_FRAME", "hesai_lidar")
    mapping_enabled = (
        os.environ.get("COLLIE_NAV2_MAPPING_ENABLED", "1") == "1"
    )
    lidar_transform = [
        "--x",
        os.environ.get("COLLIE_LIDAR_X", "0.0"),
        "--y",
        os.environ.get("COLLIE_LIDAR_Y", "0.0"),
        "--z",
        os.environ.get("COLLIE_LIDAR_Z", "0.0"),
        "--roll",
        os.environ.get("COLLIE_LIDAR_ROLL", "0.0"),
        "--pitch",
        os.environ.get("COLLIE_LIDAR_PITCH", "0.0"),
        "--yaw",
        os.environ.get("COLLIE_LIDAR_YAW", "0.0"),
        "--frame-id",
        "base_link",
        "--child-frame-id",
        lidar_frame,
    ]

    sensor_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="collie_lidar_mount",
        arguments=lidar_transform,
        output="screen",
    )
    rtab_parameters = {
        "use_sim_time": False,
        "frame_id": "base_link",
        "map_frame_id": "map",
        "database_path": os.environ.get(
            "COLLIE_NAV2_MAP_DB", "/maps/collie-rtabmap.db"
        ),
        "subscribe_depth": False,
        "subscribe_rgb": False,
        # Build the planar navigation map from the height/range-filtered XT16
        # scan. Feeding RTAB-Map the raw 3D cloud also maps returns from Woof's
        # own body inside the Nav2 footprint.
        "subscribe_scan": True,
        "subscribe_scan_cloud": False,
        "approx_sync": True,
        "sync_queue_size": 30,
        "wait_for_transform": 0.5,
        "qos_scan": 2,
        "qos_odom": 1,
        "Reg/Strategy": "1",
        "Reg/Force3DoF": "true",
        "Icp/VoxelSize": "0.05",
        "Icp/PointToPlane": "true",
        "Icp/MaxCorrespondenceDistance": "0.30",
        "Icp/Epsilon": "0.001",
        "RGBD/ProximityBySpace": "true",
        "RGBD/NeighborLinkRefining": "true",
        "RGBD/AngularUpdate": "0.04",
        "RGBD/LinearUpdate": "0.04",
        "Grid/Sensor": "0",
        "Grid/RayTracing": "true",
        "Grid/3D": "false",
        "Grid/NormalsSegmentation": "false",
        "RGBD/CreateOccupancyGrid": "true",
        "Grid/CellSize": "0.05",
        "Grid/RangeMax": "6.0",
        "Grid/MaxGroundHeight": "0.10",
        "Grid/MaxObstacleHeight": "1.20",
        "map_always_update": True,
        "Mem/IncrementalMemory": "true",
        "Mem/InitWMWithAllNodes": "true",
        "Rtabmap/DetectionRate": "2.0",
    }
    rtabmap = Node(
        package="rtabmap_slam",
        executable="rtabmap",
        namespace="rtabmap",
        name="rtabmap",
        parameters=[rtab_parameters],
        remappings=[
            ("scan", "/scan"),
            ("odom", "/odom"),
            ("map", "/map"),
        ],
        output="screen",
    )

    common = [nav2_params, {"use_sim_time": False}]
    managed = [
        "controller_server",
        "smoother_server",
        "planner_server",
        "behavior_server",
        "bt_navigator",
    ]
    nav2_nodes = [
        Node(
            package="nav2_controller",
            executable="controller_server",
            name="controller_server",
            parameters=common,
            output="screen",
        ),
        Node(
            package="nav2_smoother",
            executable="smoother_server",
            name="smoother_server",
            parameters=common,
            output="screen",
        ),
        Node(
            package="nav2_planner",
            executable="planner_server",
            name="planner_server",
            parameters=common,
            output="screen",
        ),
        Node(
            package="nav2_behaviors",
            executable="behavior_server",
            name="behavior_server",
            parameters=common,
            output="screen",
        ),
        Node(
            package="nav2_bt_navigator",
            executable="bt_navigator",
            name="bt_navigator",
            parameters=[
                *common,
                {"default_nav_to_pose_bt_xml": navigation_tree},
            ],
            output="screen",
        ),
        Node(
            package="nav2_lifecycle_manager",
            executable="lifecycle_manager",
            name="lifecycle_manager_navigation",
            parameters=[
                {
                    "use_sim_time": False,
                    "autostart": True,
                    "node_names": managed,
                }
            ],
            output="screen",
        ),
    ]
    gateway = Node(
        package="collie_nav2",
        executable="gateway",
        name="collie_nav2_gateway",
        output="screen",
    )
    foxglove = Node(
        package="foxglove_bridge",
        executable="foxglove_bridge",
        name="collie_foxglove",
        parameters=[{"port": 8767, "address": "0.0.0.0"}],
        output="screen",
    )

    actions = [sensor_tf, gateway, foxglove]
    if mapping_enabled:
        actions.append(TimerAction(period=4.0, actions=[rtabmap]))
    actions.append(TimerAction(period=7.0, actions=nav2_nodes))
    return LaunchDescription(actions)
