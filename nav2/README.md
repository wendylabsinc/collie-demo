# Collie Nav2 return service

This companion Wendy app gives `collie-demo` a map-frame `NavigateToPose`
return backend. It follows the bridge/SLAM/Nav2 layering used by
`autonomous-go2-inspection`, but keeps the existing `collie-demo` motion
adapter as the only Unitree motion owner.

## Motion ownership

`Nav2 /cmd_vel` is never published to Unitree DDS directly:

1. The read-only bridge republishes odometry and the selected point cloud from
   raw ROS domain 0 to standard ROS domain 30.
2. The XT16 cloud is height/range-filtered into `/scan`; RTAB-Map consumes
   that planar scan so floor and robot-body returns cannot seed the map.
   RTAB-Map owns `map -> odom`; Nav2 plans to the saved Home pose.
3. The gateway relays bounded forward/yaw heartbeats to
   `http://127.0.0.1:8096/api/navigation/*`.
4. `collie-demo` owns the exclusive direct-Sport Unitree lease, command
   watchdog, and hardware STOP. Nav2 costmaps are the collision planner for
   this lease; the fruit-approach factory-avoidance path is not shared.

The gateway rejects reverse, lateral motion, forward commands above 0.30 m/s,
yaw above 0.50 rad/s, stale commands, stale localization, stale LiDAR, stale
maps, stalled progress, and goals outside the three-metre stage envelope.
During an intentional Nav2 recovery transition, a command older than 350 ms is
replaced with an explicit zero heartbeat rather than replayed. This keeps the
independent hardware watchdog satisfied while Woof is stopped; a gap longer
than three seconds still cancels the goal and revokes motion ownership.
Its forward-only behavior tree can clear costmaps, spin, or wait, but contains
no backup recovery. Path following uses the regulated pure-pursuit controller
with reversing disabled and collision detection enabled. The external result
gate remains 0.10 m and 5 degrees; Nav2's internal goal checker uses 0.08 m
and 4 degrees to leave verification margin.

## Hesai XT16 gate

The preferred input is `/hesai/points` in frame `hesai_lidar`. Mapping can be
started with a provisional mount transform for inspection, but the gateway
will report `ready=false` and refuse Home capture or navigation until
`COLLIE_LIDAR_TRANSFORM_VALIDATED=1`.

Do not set that flag merely because the map looks plausible. First verify the
physical `base_link -> hesai_lidar` translation and rotation against a level
floor, vertical walls, and a known obstacle while Woof remains stationary.

The bridge image includes a read-only dual-cloud capture utility. It compares
the external XT16 with `/utlidar/cloud_base`, which the Go2 publishes in
`base_link`. Capture at least three independent stationary samples:

```bash
wendy --device woof.local device attach collie-nav2_bridge -- \
  python3 /app/tools/capture_extrinsic_pair.py \
  --frames 20 --minimum-duration 3 \
  --output /tmp/extrinsic-01.npz
```

Copy each compressed capture to the workstation and estimate consistency:

```bash
python3 -m pip install -r nav2/tools/requirements-calibration.txt
python3 nav2/tools/estimate_hesai_extrinsic.py \
  /path/to/extrinsic-01.npz \
  /path/to/extrinsic-02.npz \
  /path/to/extrinsic-03.npz \
  --output /path/to/extrinsic-report.json
```

The report deliberately emits `motion_unlock_authorized=false` even when its
numerical gates pass. Review transformed floor alignment, vertical structure,
and a known obstacle in Foxglove before entering the six values in
`nav2/wendy.json` and changing the validation flag.

For a read-only Hesai preview:

```bash
cd /Users/smile/go2-follow-demo
NAV_RUNTIME_MODE=sensor-only \
  wendy --device woof.local run --service nav --detach -y

cd /Users/smile/Documents/collie-demo
wendy --device woof.local run --prefix nav2 --detach -y
```

The preview values are intentionally explicit in `nav2/wendy.json`: Hesai
input, a measured transform candidate, and validation flag `0`. On 2026-07-30,
three independent stationary captures against `/utlidar/cloud_base` produced:

- translation mean `(0.247072, -0.052732, 0.104320)` m, with worst-axis
  standard deviation `0.00794` m;
- roll/pitch/yaw mean `(-0.3585, 0.7531, 90.1512)` degrees, with worst-axis
  standard deviation `0.232` degrees;
- floor angular error `0.118-0.393` degrees after refinement;
- onboard non-floor geometry coverage `88.2-89.7%` within `0.18` m, with
  `0.126-0.131` m 90th-percentile residual.

Those values are a calibration candidate, not motion authorization. Validate
the fresh map, transformed floor/walls, and a known obstacle before changing
the flag to `1`. Shell variables in front of `wendy run` do not override
manifest service environment, so do not use them as a substitute for that
review.

Inspect:

- gateway: `http://woof.local:8100/api/health`
- Foxglove: `ws://woof.local:8767`
- expected topics: `/odom`, `/pointcloud`, `/scan`, `/map`, `/tf`,
  `/plan`, `/local_costmap/costmap`, `/global_costmap/costmap`

Port `8098` remains reserved for the existing Collie voice service; the Nav2
gateway uses `8100`.

Once the transform and read-only stack pass validation, set
`COLLIE_RETURN_BACKEND=nav2` in the main app image and redeploy both apps.
That switches `collie-demo` navigation commands to its exclusive direct Sport
lease; Nav2 costmaps become the sole collision planner, and the gateway/runtime
watchdogs remain independent fail-stop boundaries.

## Hardware-free integration trial

The autonomy image includes a local-only planar stage that exercises the real
Nav2 planner, gateway, HTTP motion relay, map-frame Home capture, final result
gate, and STOP. It does not import or address the Unitree SDK.

```bash
docker build -t collie-nav2-autonomy:test nav2/autonomy
docker run -d --name collie-nav2-integration \
  -e COLLIE_LIDAR_TRANSFORM_VALIDATED=1 \
  -e COLLIE_NAV2_MAPPING_ENABLED=0 \
  collie-nav2-autonomy:test

docker exec -t collie-nav2-integration bash -lc \
  'source /opt/ros/jazzy/setup.bash &&
   source /nav2_ws/install/setup.bash &&
   export CYCLONEDDS_URI=file:///tmp/collie-nav2-domain.xml &&
   python3 /integration/synthetic_return_trial.py --scenario success'
```

Repeat the final command with `detour`, `localization-loss`, `no-path`, and
`stalled-motion`. The detour scenario must route around a finite wall and
still satisfy the final pose contract. Each failure scenario passes only after
the emulated motion owner confirms `armed=false` and at least one STOP. The
no-path trial additionally requires zero forward commands before failure.
The integration-only mapping override prevents RTAB-Map from replacing the
synthetic occupancy grid and `map -> odom` transform; deployed Woof keeps
`COLLIE_NAV2_MAPPING_ENABLED=1`.
