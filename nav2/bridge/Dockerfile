FROM ros:jazzy-ros-base

SHELL ["/bin/bash", "-lc"]
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      iproute2 tini python3-scipy \
      ros-jazzy-pointcloud-to-laserscan \
      ros-jazzy-rmw-cyclonedds-cpp \
      ros-jazzy-sensor-msgs-py \
      ros-jazzy-tf2-ros \
    && rm -rf /var/lib/apt/lists/*

COPY bridge.py /app/bridge.py
COPY tools /app/tools
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV GO2_ROS_DOMAIN_ID=0 \
    BRIDGE_ROS_DOMAIN_ID=30 \
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    GO2_IP=192.168.123.161 \
    LIDAR_SOURCE_TOPIC=/hesai/points \
    RAW_SCAN_TOPIC=/collie/raw_scan \
    SCAN_OUTPUT_TOPIC=/scan \
    ODOM_SOURCE_TOPIC=/utlidar/robot_odom \
    ODOM_OUTPUT_TOPIC=/odom

ENTRYPOINT ["/usr/bin/tini", "--", "/entrypoint.sh"]
