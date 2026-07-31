#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/jazzy/setup.bash
set -u

LOCAL_IP="$(ip -4 route get "${GO2_IP:-192.168.123.161}" 2>/dev/null \
  | awk '{for (i=1;i<=NF;i++) if ($i=="src") {print $(i+1); exit}}')"

export CYCLONEDDS_FILE=/tmp/collie-nav2-cyclonedds.xml
cat > "${CYCLONEDDS_FILE}" <<XML
<?xml version="1.0" encoding="UTF-8"?>
<CycloneDDS xmlns="https://cdds.io/config">
  <Domain Id="${GO2_ROS_DOMAIN_ID:-0}">
    <General>
      <Interfaces>
        <NetworkInterface address="${LOCAL_IP:-auto}" priority="default" multicast="default" />
      </Interfaces>
      <AllowMulticast>true</AllowMulticast>
      <MaxMessageSize>65500B</MaxMessageSize>
    </General>
    <Discovery>
      <ParticipantIndex>auto</ParticipantIndex>
      <MaxAutoParticipantIndex>60</MaxAutoParticipantIndex>
    </Discovery>
  </Domain>
  <Domain Id="${BRIDGE_ROS_DOMAIN_ID:-30}">
    <General>
      <AllowMulticast>true</AllowMulticast>
      <EnableMulticastLoopback>true</EnableMulticastLoopback>
      <MaxMessageSize>65500B</MaxMessageSize>
    </General>
    <Discovery>
      <ParticipantIndex>auto</ParticipantIndex>
      <MaxAutoParticipantIndex>60</MaxAutoParticipantIndex>
      <Peers><Peer Address="127.0.0.1" /></Peers>
    </Discovery>
  </Domain>
</CycloneDDS>
XML
export CYCLONEDDS_URI="file://${CYCLONEDDS_FILE}"

echo "Starting native XT16 scan filter: lidar=${LIDAR_SOURCE_TOPIC:-/hesai/points}"
ROS_DOMAIN_ID="${GO2_ROS_DOMAIN_ID:-0}" ros2 run tf2_ros static_transform_publisher \
  --x "${COLLIE_LIDAR_X:-0.0}" \
  --y "${COLLIE_LIDAR_Y:-0.0}" \
  --z "${COLLIE_LIDAR_Z:-0.0}" \
  --roll "${COLLIE_LIDAR_ROLL:-0.0}" \
  --pitch "${COLLIE_LIDAR_PITCH:-0.0}" \
  --yaw "${COLLIE_LIDAR_YAW:-0.0}" \
  --frame-id base_link \
  --child-frame-id "${COLLIE_LIDAR_FRAME:-hesai_lidar}" &
TF_PID=$!

ROS_DOMAIN_ID="${GO2_ROS_DOMAIN_ID:-0}" ros2 run \
  pointcloud_to_laserscan pointcloud_to_laserscan_node \
  --ros-args \
  -r "cloud_in:=${LIDAR_SOURCE_TOPIC:-/hesai/points}" \
  -r "scan:=${RAW_SCAN_TOPIC:-/collie/raw_scan}" \
  -p target_frame:=base_link \
  -p transform_tolerance:=0.25 \
  -p min_height:=0.05 \
  -p max_height:=0.90 \
  -p angle_min:=-3.14159 \
  -p angle_max:=3.14159 \
  -p angle_increment:=0.00873 \
  -p scan_time:=0.10 \
  -p range_min:=0.35 \
  -p range_max:=8.0 \
  -p use_inf:=true \
  -p inf_epsilon:=1.0 &
SCAN_PID=$!

cleanup() {
  kill "${BRIDGE_PID:-}" "${SCAN_PID}" "${TF_PID}" 2>/dev/null || true
  wait "${BRIDGE_PID:-}" "${SCAN_PID}" "${TF_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Starting read-only Go2 sensor bridge: scan=${RAW_SCAN_TOPIC:-/collie/raw_scan}"
python3 /app/bridge.py &
BRIDGE_PID=$!
wait -n "${BRIDGE_PID}" "${SCAN_PID}" "${TF_PID}"
