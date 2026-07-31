# Collie demo repository map

This document ties the intended Border Collie routine to the code that is
currently deployed or available in this repository. It is the starting point
for return-home and camera-stability work.

## Intended routine versus current implementation

| Intended step | Current implementation | Gap or risk |
| --- | --- | --- |
| Face the person | No mission phase or person detector does this. The round begins from whichever pose Woof already has when Home is captured. | Implement an explicit person-facing/alignment phase, or define this as an operator setup requirement. |
| Hear “go to pear” | `voice/commands.py` accepts a small allowlist; `voice/main.py` calls `POST /api/voice/mission`. The UI can submit the same typed fruit command. | Speech currently accepts bare fruit names and `Find ...`; the desired phrase and interaction should be specified and tested explicitly. |
| Turn toward the fruit | `CollieRuntime._run_measured_turn()` commands a bounded yaw and verifies it with fresh Go2 odometry. | This path exists and has simulation tests, but still needs a repeatable hardware acceptance run. |
| Find and approach the fruit | `_search_for_memory()`, `_wait_for_demo_go()`, `_start_memory_approach()`, and `_monitor_memory_approach()` use fresh YOLO detections and the guarded motion adapter. | Camera stalls can invalidate the target or abort the mission before/during this stage. |
| Sit and bark | `_rest_at_target()` uses Unitree `StandDown`, holds for five seconds, then uses `StandUp` and `BalanceStand`. The voice mission monitor barks when rest status becomes `holding`. | Bark is coordinated by the voice service rather than the core mission, so loss of that service can produce a silent rest without aborting motion safety. |
| Turn around and return to start | `_return_home()` captures Home before the outbound turn, reorients toward it, drives to the saved pose, and restores the saved heading. It supports `local_odometry` and `nav2`. | The code exists, but the default image selects `nav2` while the root `wendy.json` does not launch the separate `nav2/wendy.json` app. A root-only deployment therefore cannot complete the default return path. Neither backend is yet documented as hardware-qualified for this exact full routine. |

## Active runtime boundary

The root Wendy app starts two services:

- `app`: `collie_demo.supervisor` -> `collie_demo.main` -> FastAPI in
  `collie_demo.app` -> the long-lived `CollieRuntime` state machine.
- `voice`: one Go2 WebRTC peer for microphone, camera frames, and AudioHub bark.
  It publishes fresh JPEGs through `voice/camera_broker.py` and sends only
  allowlisted mission commands to the core app.

The main data/control flow is:

```text
Go2 WebRTC video -> voice camera broker -> BrokerCamera -> YOLO worker
                                                     -> browser stream

typed/speech fruit -> voice service -> core mission state machine
                                     -> exclusive motion adapter -> Go2

Go2 odometry -> Home capture -> measured departure turn
                            -> local odometry return OR Nav2 return client
```

`nav2/` is a separate Wendy app, not a service in the root manifest. It bridges
Hesai/odometry data into ROS 2, owns mapping and path planning, then relays
bounded forward/yaw commands back through the core app. It must be deployed and
healthy separately whenever `COLLIE_RETURN_BACKEND=nav2`.

## Camera-freeze boundary

The current design already contains the right fail-closed concepts:

- the voice process is the sole native WebRTC peer;
- every reconnect rotates a camera generation and clears buffered bytes;
- broker frames carry source timestamps and become unusable after 750 ms;
- the runtime invalidates an old target when the generation changes;
- YOLO runs in a separate worker so inference cannot block motion heartbeats;
- camera and target freshness are part of `stage_ready`.

What is missing is physical qualification and actionable evidence. Before
changing motion logic, run a 30-minute camera soak plus forced peer-loss tests
and record broker generation, source-frame age, request latency, inference age,
mission phase, and reconnect reason. This will distinguish a frozen WebRTC
producer from a blocked broker request, slow decode, slow inference, or browser
display lag.

## Repository areas

- `src/collie_demo/`: core state machine, camera adapters, detector, safety
  boundary, pointing experiment, and both return clients.
- `voice/`: deployed voice/WebRTC/camera-broker service.
- `web/`: deployed operator UI.
- `nav2/`: separately deployed mapped-return companion.
- `thor-audio/`: optional desk-side microphone/speaker companion referenced by
  the voice service; it is not part of the root Woof deployment.
- `models/pointing/`: the one hash-pinned actor used by the guarded manual
  pointing experiment.
- `tools/`: documented detector build and benchmark utilities.
- `artifacts/debug/`: retained hardware-run evidence, not runtime input.

The local webcam/UI entry points and the pointing diagnostics are not part of
the autonomous stage path, but they remain documented operator/development
tools and should not be deleted as dead runtime code.

## Cleanup completed in the first pass

- Removed the unreachable reverse-clearance return controller and its private
  reverse-motion API. The active return has already moved to measured
  reorientation followed by forward-only translation.
- Removed the six unused reverse-clearance settings and misleading clearance
  telemetry.
- Removed the obsolete 49-value pointing-policy helpers and old model artifact;
  the active guarded actor consumes the documented 47-value contract.
- Removed unused compatibility helpers, imports, locals, and geometry helpers.

## Next implementation slices

1. Make the deployment contract unambiguous: either include the Nav2 companion
   in the deployment procedure and readiness gate, or select
   `local_odometry` until Nav2 is actually deployed and hardware-qualified.
2. Add a camera soak/probe that records the telemetry above and survives long
   enough to identify the first stale boundary.
3. Add a hardware acceptance checklist for Home capture, departure turn,
   forward return, final heading, stop behavior, and camera reconnect.
4. Only after those measurements, adjust the return controller or camera
   ownership/reconnect policy.
