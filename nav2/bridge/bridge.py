"""Read-only Go2 sensor bridge for the Collie Nav2 stack.

Raw Go2/Hesai ROS topics remain on domain 0. This process republishes only
standard sensor and odometry messages on domain 30. It has no Unitree command
publisher and therefore cannot move Woof.
"""

from __future__ import annotations

import os
import time
from threading import Lock, Thread

import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformBroadcaster


def context_for(domain_id: int) -> Context:
    context = Context()
    rclpy.init(context=context, domain_id=domain_id)
    return context


class SensorSource(Node):
    def __init__(self, context: Context, sink: "SensorSink") -> None:
        super().__init__("collie_sensor_source", context=context)
        self.sink = sink
        state_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        scan_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        # The XT16 cloud is converted to a compact LaserScan by the native
        # pointcloud_to_laserscan process before it reaches this Python
        # cross-domain bridge. Relaying the raw ~9 MB/s cloud here consumed a
        # CPU core and starved Unitree's camera RPC on the four-core Go2.
        # The callbacks now only retimestamp and relay odometry/scan messages,
        # so one executor avoids the Python thread-pool overhead that used to
        # compete with the camera process.
        self.create_subscription(
            Odometry,
            os.environ.get("ODOM_SOURCE_TOPIC", "/utlidar/robot_odom"),
            self.sink.publish_odom,
            state_qos,
        )
        self.create_subscription(
            LaserScan,
            os.environ.get("RAW_SCAN_TOPIC", "/collie/raw_scan"),
            self.sink.publish_scan,
            scan_qos,
        )
        self.get_logger().info(
            "Read-only source subscriptions active on raw domain "
            f"{os.environ.get('GO2_ROS_DOMAIN_ID', '0')}"
        )


class SensorSink(Node):
    def __init__(self, context: Context) -> None:
        super().__init__("collie_sensor_sink", context=context)
        state_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        # A stale LiDAR backlog is worse than a dropped scan for collision
        # planning. Keep only the two newest scans and match ROS sensor
        # best-effort semantics end to end.
        scan_qos = QoSProfile(
            depth=2,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.odom_pub = self.create_publisher(
            Odometry,
            os.environ.get("ODOM_OUTPUT_TOPIC", "/odom"),
            state_qos,
        )
        self.scan_pub = self.create_publisher(
            LaserScan,
            os.environ.get("SCAN_OUTPUT_TOPIC", "/scan"),
            scan_qos,
        )
        self.tf = TransformBroadcaster(self)
        self.odom_count = 0
        self.scan_count = 0
        self.odom_dropped = 0
        self._counter_lock = Lock()
        self._last_report_s = time.monotonic()
        self._last_report_counts = (0, 0)
        self._odom_min_period_s = 1.0 / max(
            1.0,
            float(os.environ.get("ODOM_OUTPUT_MAX_HZ", "50")),
        )
        self._last_odom_s = 0.0
        self.create_timer(5.0, self._report_rates)

    def publish_odom(self, msg: Odometry) -> None:
        now_s = time.monotonic()
        if now_s - self._last_odom_s < self._odom_min_period_s:
            with self._counter_lock:
                self.odom_dropped += 1
            return
        self._last_odom_s = now_s
        stamp = self.get_clock().now().to_msg()
        msg.header.stamp = stamp
        msg.header.frame_id = "odom"
        msg.child_frame_id = "base_link"
        self.odom_pub.publish(msg)

        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = "odom"
        transform.child_frame_id = "base_link"
        transform.transform.translation.x = msg.pose.pose.position.x
        transform.transform.translation.y = msg.pose.pose.position.y
        transform.transform.translation.z = msg.pose.pose.position.z
        transform.transform.rotation = msg.pose.pose.orientation
        self.tf.sendTransform(transform)
        with self._counter_lock:
            self.odom_count += 1
            first = self.odom_count == 1
        if first:
            self.get_logger().info("Received first Go2 odometry sample")

    def publish_scan(self, msg: LaserScan) -> None:
        msg.header.stamp = self.get_clock().now().to_msg()
        self.scan_pub.publish(msg)
        with self._counter_lock:
            self.scan_count += 1
            first = self.scan_count == 1
        if first:
            self.get_logger().info(
                f"Received first filtered scan in frame {msg.header.frame_id!r}"
            )

    def _report_rates(self) -> None:
        now_s = time.monotonic()
        duration_s = max(1e-9, now_s - self._last_report_s)
        with self._counter_lock:
            counts = (self.odom_count, self.scan_count)
            dropped_odom = self.odom_dropped
        previous = self._last_report_counts
        rates = tuple(
            (current - old) / duration_s
            for current, old in zip(counts, previous)
        )
        self.get_logger().info(
            "Bridge output rates: "
            f"odom={rates[0]:.1f}Hz scan={rates[1]:.1f}Hz "
            f"dropped_odom={dropped_odom}"
        )
        self._last_report_s = now_s
        self._last_report_counts = counts


def main() -> None:
    source_context = context_for(int(os.environ.get("GO2_ROS_DOMAIN_ID", "0")))
    sink_context = context_for(int(os.environ.get("BRIDGE_ROS_DOMAIN_ID", "30")))
    sink = SensorSink(sink_context)
    source = SensorSource(source_context, sink)
    source_executor = SingleThreadedExecutor(context=source_context)
    sink_executor = SingleThreadedExecutor(context=sink_context)
    source_executor.add_node(source)
    sink_executor.add_node(sink)
    sink_thread = Thread(target=sink_executor.spin, daemon=True)
    sink_thread.start()
    try:
        source_executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        source_executor.shutdown()
        sink_executor.shutdown()
        source.destroy_node()
        sink.destroy_node()
        rclpy.try_shutdown(context=source_context)
        rclpy.try_shutdown(context=sink_context)
        sink_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
