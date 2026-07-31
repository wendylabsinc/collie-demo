FROM ros:jazzy-ros-base

SHELL ["/bin/bash", "-lc"]
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      iproute2 tini python3-colcon-common-extensions python3-pip \
      ros-jazzy-rmw-cyclonedds-cpp \
      ros-jazzy-rtabmap-ros \
      ros-jazzy-navigation2 ros-jazzy-nav2-bringup \
      ros-jazzy-pointcloud-to-laserscan \
      ros-jazzy-tf2-ros ros-jazzy-tf2-geometry-msgs \
      ros-jazzy-foxglove-bridge \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache-dir --break-system-packages \
      "fastapi>=0.115,<1" "httpx>=0.28,<1" "uvicorn>=0.34,<1"

COPY ros2_ws/src /nav2_ws/src
RUN source /opt/ros/jazzy/setup.bash \
    && cd /nav2_ws \
    && colcon build \
    && rm -rf build log

COPY integration /integration
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV ROS_DOMAIN_ID=30 \
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    COLLIE_NAV2_PORT=8100 \
    COLLIE_RUNTIME_URL=http://127.0.0.1:8096 \
    COLLIE_NAV2_MAP_DB=/maps/collie-rtabmap.db \
    COLLIE_NAV2_MAPPING_ENABLED=1 \
    COLLIE_LIDAR_FRAME=hesai_lidar \
    COLLIE_LIDAR_X=0.0 \
    COLLIE_LIDAR_Y=0.0 \
    COLLIE_LIDAR_Z=0.0 \
    COLLIE_LIDAR_ROLL=0.0 \
    COLLIE_LIDAR_PITCH=0.0 \
    COLLIE_LIDAR_YAW=0.0 \
    COLLIE_LIDAR_TRANSFORM_VALIDATED=0

EXPOSE 8100 8767
ENTRYPOINT ["/usr/bin/tini", "--", "/entrypoint.sh"]
