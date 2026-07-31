#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/jazzy/setup.bash
source /nav2_ws/install/setup.bash
set -u

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-30}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
mkdir -p "$(dirname "${COLLIE_NAV2_MAP_DB:-/maps/collie-rtabmap.db}")"
export CYCLONEDDS_FILE=/tmp/collie-nav2-domain.xml
cat > "${CYCLONEDDS_FILE}" <<XML
<?xml version="1.0" encoding="UTF-8"?>
<CycloneDDS xmlns="https://cdds.io/config">
  <Domain Id="${ROS_DOMAIN_ID}">
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

if [[ "${COLLIE_LIDAR_FRAME:-hesai_lidar}" == "hesai_lidar" \
      && "${COLLIE_LIDAR_TRANSFORM_VALIDATED:-0}" != "1" ]]; then
  echo "Hesai transform is provisional: Nav2 will map but the gateway will refuse motion."
fi

exec ros2 launch collie_nav2 bringup.launch.py
