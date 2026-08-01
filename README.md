# Collie Demo

See [`docs/repository-map.md`](docs/repository-map.md) for the intended Border
Collie routine, the code path for each step, current deployment gaps, and the
cleanup/verification plan. Completed hardware tests and their visual evidence
are tracked in [`docs/validation-results.md`](docs/validation-results.md).

A self-contained Wendy app for Woof that can save a locally detected fruit class,
turn around, recognize a fresh fruit of that class, safely approach it, and return
to its saved start pose. The
validated manual apple/banana/pear follower remains available as a fallback. It uses a
YOLOE-11m checkpoint whose visual prompt embeddings were baked from the three
physical stage props. The Woof image runs a TensorRT 10.7 FP16 engine exported
on its Orin; the original PyTorch checkpoint remains in the image as the source
artifact.
The voice service owns one native Go2 WebRTC peer and shares its video frames
through a local, freshness-bounded camera broker while using the same peer for
the Go2 microphone and native bark. Inference, annotation, motion supervision,
and the browser UI all run locally in robot containers. No Roboflow service,
hosted inference API, Hugging Face key, or internet connection is used for
vision inference. The optional voice service streams microphone PCM to
ElevenLabs Scribe v2 Realtime; motion control and fruit inference remain on
Woof.

The audience-facing page at `/` is intentionally limited to **Activate Demo**
and **Stop Now**. Detailed stage controls and telemetry live at `/debug`.
Every activation records compact preflight, camera, detector, voice, mission,
and return-home snapshots in browser-local run history, which the debug page
can display or export as JSON. Reusable single-purpose operator tools live
under `/tests`; legacy calibration URLs remain as compatibility aliases.

An optional companion stack in [`nav2/`](nav2/README.md) bridges Woof sensors
read-only into ROS domain 30, builds a map with RTAB-Map, and plans the return
with Nav2. The stage image now selects that backend by default, while the
separate gateway remains fail-closed until the real Hesai mount transform and
live mapping pipeline are qualified on Woof.

## Full demo sequence

The canonical audience-facing demo is:

1. **Verify preflight.** Confirm that the camera, fruit detector, motion
   controller, microphone, and configured return-home backend report ready.
2. **Face the person.** Position Woof facing the person who will give the
   command. Automatic person-facing alignment is not implemented yet.
3. **Wait for a command.** Woof remains stopped and listens.
4. **Request a fruit.** Say an allowlisted fruit command such as `pear`, or type
   the fruit into the UI. Exact `Go to ...` phrase support is planned but is not
   currently part of the voice allowlist.
5. **Acknowledge the command.** Woof barks to confirm the requested fruit.
6. **Capture Home.** Woof records its starting position and heading. The Nav2
   backend also captures a stable map-frame Home pose.
7. **Turn toward the search area.** Woof performs a measured approximately
   180-degree turn.
8. **Search for the fruit.** If the requested fruit is not visible, Woof runs a
   bounded rotational search.
9. **Confirm the fruit.** Multiple fresh detector results must agree on the
   requested class before approach motion is allowed.
10. **Release Go.** Voice and typed full-sequence missions release Go
    automatically. The manual remember-and-find workflow waits for the
    operator's **Go to Fruit** button.
11. **Walk to the fruit.** Woof steers toward the fresh target while monitoring
    camera age, target age, obstacle avoidance, and motion watchdogs.
12. **Confirm arrival.** Repeated lower-camera detections establish that Woof is
    close. When that confirmed fruit leaves the bottom of the view, Woof sends
    one 1.0 m/s forward push for 0.4 seconds.
13. **Stop at the fruit.** All walking commands stop and the active motion lease
    is released.
14. **Lie down and bark.** Woof runs Unitree `StandDown`, and the voice service
    plays the arrival bark while the rest status is `holding`.
15. **Hold the pose.** Woof remains down for approximately five seconds.
16. **Stand for the return.** Woof runs `StandUp` followed by `BalanceStand` so
    locomotion is active again.
17. **Stabilize localization.** The runtime waits for fresh, stationary
    odometry before starting the return leg.
18. **Turn toward Home.** Woof calculates the direction of the saved start pose
    and performs a measured departure turn.
19. **Walk back to Home.** Nav2 follows a mapped obstacle-aware path; the
    diagnostic local-odometry backend follows a bounded forward-only route.
20. **Restore the original heading.** At the saved position, Woof turns to
    match its original orientation and face the starting direction again.
21. **Report success and reset.** The UI reports success, voice returns to
    listening, and the demo is ready for the next fruit command.

`STOP NOW`, a stop voice command, stale camera data, target loss, localization
loss, or a motion-watchdog failure must stop Woof during any movement phase.

## Measured forward-command deadband

`MOTION-DEADBAND-001` measured Woof's forward response through Unitree's
factory `ObstaclesAvoidClient` remote-command path using 0.4-second,
forward-only pulses. The operator observed no physical movement at 0.25 m/s
and physical movement at both 0.50 m/s and 1.00 m/s. Until that path is
recalibrated, treat **0.50 m/s as Woof's minimum reliable forward command on
this interface**.

This is a measured actuation threshold, not a desired cruising speed. Commands
below it can be accepted and renewed without producing useful travel. Direct
`SportClient` calibration subsequently confirmed physical steps at both
**0.25 m/s** and **0.50 m/s**. The Nav2 controller now uses **0.25 m/s as its
provisional positive forward-command floor**; zero and rotation-only commands
are not promoted. Values below 0.25 m/s remain to be tested before calling
0.25 m/s the exact direct-path deadband boundary.
Evidence is recorded in
[`docs/validation-results.md`](docs/validation-results.md#motion-deadband-001--factory-avoidance-forward-command-deadband).

## Current behavior

- Bundles the local PyTorch source checkpoint and Woof-specific TensorRT FP16
  engine; runtime inference does not call a hosted API.
- Detects only the three visual-prompt classes: apple, banana, and pear.
- Bundles the hash-pinned `locked_point_v19` iteration-73,000 TorchScript
  balance actor and exposes a guarded browser flow. The actor consumes 47
  values and controls nine support joints while the front-right point follows
  its deterministic training schedule. It receives the exact normalized
  selected `xyxy` box plus live Go2 proprioception and cannot silently switch
  to another detection.
- Keeps the pointing runner behind the same single-motion-owner boundary as
  following and navigation. The UI exposes Stop, target changes are rejected
  during low-level control, target loss aborts, and every exit path attempts to
  return to the captured StandDown pose and restore Sport mode.
- Retains the 0.30 rad roll, 0.35 rad pitch, 22 Nm estimated-torque,
  4 rad/s joint-speed, 0.60 rad/s target-rate, low-state freshness, policy
  checksum, unchanged target-lock, and 800 ms bbox-age guards. The stage runner
  contains no roll-guard bypass. The six-second skill includes one second of
  setup and a one-second deterministic point ramp.
- Runs a separate persistent Go2 WebRTC microphone service on port 8098. It
  accepts only deterministic bare `apple|banana|pear` commands, the legacy
  `Find [the] ...` form, plus `stop|abort|cancel`; arbitrary transcripts can
  never become motor commands.
- Uses that voice service as the sole native Go2 WebRTC peer for video, mic,
  and AudioHub bark. The main runtime consumes broadcast JPEG snapshots from
  the local broker; it never opens Unitree `VideoClient` in the stage profile.
  A reconnect rotates the stream generation, clears old frames, aborts any
  active mission, and requires a new fruit command before motion can resume.
- A committed bare `apple`, `banana`, or `pear` command (with `pair` accepted
  as a Scribe homophone for pear) plays the user-supplied native AudioHub bark,
  stores only the requested YOLO class, and starts the normal guarded mission.
  The voice-only path skips the initial Hello and recognition Stretch gestures.
  The voice bridge automatically releases Go only after the mission reports a
  fresh multi-frame class lock and all stage-health checks remain ready.
- The autonomous arrival uses Unitree's stock `StandDown`, holds for five
  seconds, and calls `StandUp` before returning Home. The experimental
  standing-point policy remains available only through its manual UI panel and
  is not part of the full fruit sequence.
- Once repeated detections confirm a near fruit and it then leaves the lower
  camera edge, a final time-bounded direct-path push runs at 1.0 m/s for 0.4
  seconds. Woof sends zero immediately afterward and continues into the same
  StandDown sequence. Merely seeing the fruit nearby does not trigger the
  push, and camera or pose staleness still aborts.
- Loads the Scribe credential from the root-only Wendy persistent volume at
  `/state/elevenlabs.env`. The API key is never baked into an image, committed,
  returned by `/api/status`, or printed to logs.
- Runs the Orin-specific `collie-fruit-yoloe11m.engine` with the model task
  explicitly set to `segment`; this is required because a serialized TensorRT
  engine cannot reliably infer its Ultralytics task from the filename.
- Draws labels, confidence scores, bounding boxes, and box centers.
- Prints every detection to the container log.
- Serves a low-latency Go2 MJPEG stream with browser-rendered detection boxes,
  plus the legacy annotated snapshot and structured detections on port 8096.
- Uses per-class confidence thresholds: apple 70%, banana 20%, and pear 70%.
  The lower banana threshold preserves detection as the robot closes in; the
  60-frame close-view live set scored 28.1-45.5%, the benchmark had no banana
  false positives even at 5%, and three historical non-target frames stayed
  below 6.2%.
- Makes every live detection selectable. Selection sends both the model class
  and bounding-box center, so either of two visible apples can be selected
  independently; one click on `Follow Selected Fruit` then starts the guarded
  approach loop.
- Uses each fresh GPU YOLO result as the authoritative target observation and
  collapses overlapping same-class boxes before they reach the UI or control
  loop.
- Revalidates the selected track against every new YOLO result. If the chosen
  fruit disappears or the tracker drifts to a different object for three
  consecutive results, selection is cleared and motion is stopped. One or two
  transient misses do not interrupt the approach.
- Marks tracker-only frames honestly in the UI instead of repeating an old
  YOLO confidence score as though it were current.
- Continuously steers from the latest observation of the selected object.
- Runs the follow pulse loop inside the robot service after one Follow click;
  browser timer throttling or a slow status refresh cannot interrupt command
  renewal. The independent 350 ms motion watchdog remains the final brake.
- Waits briefly for a freshly selected target to become stable and freshly
  YOLO-verified, so the operator does not need to click Follow twice.
- Rejects detections older than 750 ms and clears stale detector output on
  inference errors.
- Disarms whenever the selected object changes, so changing targets cannot
  redirect an active motion burst.
- Runs produce inference in a separate worker so a slow YOLO frame cannot
  block hold pulses, stop commands, or the independent motion watchdog.
- Keeps the exact arm confirmation, dedicated stop control, factory avoidance,
  forward time budget, target-loss stop, and `StopMove` safety boundary.
- Uses Woof's Jetson GPU for YOLO inference and reports the requested/resolved
  device, CUDA version, Torch version, and current inference latency in
  `/api/status`.
- Reports aggregate `stage_ready` health for the camera, YOLO worker, CUDA GPU,
  and motion adapter. The UI displays verification age and the current miss
  count. The out-of-process supervisor monitors process reachability; degraded
  sensor readiness remains a fail-closed motion gate inside the child runtime.
- Assigns every explicit fruit selection a process-scoped `runtime_id` and a
  monotonically increasing `target_lock_id`. Remote voice/UI clients must
  revalidate both values before starting the bounded follower.
- Preserves a disarmed user's fruit choice through brief camera or detector
  gaps while immediately removing motion readiness. The same lock can recover
  only after fresh YOLO reacquisition; an active follower still clears and
  stops on the bounded loss rule.
- Gives CUDA/model initialization a 60-second supervisor grace period so the
  cold first inference cannot create a false restart loop. A child that exits
  still fails immediately. After warm-up, 80 consecutive failed probes
  (roughly 60 seconds including request timeouts) trigger the independent
  emergency brake and restart path, while brief Jetson inference stalls only
  close the fail-safe motion readiness gate.
- `Save Class` stores the selected YOLO label and one reference crop for the UI
  while motion remains disarmed. The per-round class target is local to the robot process and is
  cleared only by `Reset Round`, a replacement capture, or service restart.
- After a successful save, Woof performs Unitree's stock `Hello` paw-forward
  gesture once. The gesture holds the exclusive motion boundary, keeps factory
  avoidance and translational control disarmed, and must finish before the
  turn/search mission can start. Set `COLLIE_INITIAL_HELLO_ENABLED=0` to disable
  the acknowledgement without changing fruit memory behavior.
- After Woof confirms the saved fruit class, it releases the search-motion lease
  and performs Unitree's stock `Stretch` once. It waits for the animation to
  settle, remains motion-disarmed, and requires several fresh detections of the saved class
  before presenting an explicit `Go to Fruit` button. Woof remains stopped and
  disarmed until that button is pressed, then requires another set of fresh
  same-class detections before it can arm the guarded approach. If the fruit
  moved or disappeared during the stretch or operator pause, the mission aborts
  instead of walking toward stale image coordinates. Stretch itself is cosmetic:
  an RPC failure is reported in mission telemetry but cannot abort a valid class
  lock. `COLLIE_SKILL_TIMEOUT_S` and `COLLIE_CLIENT_TIMEOUT_S` default to 12
  seconds so long-running stock animations are not misreported as timeouts. Set
  `COLLIE_MATCH_STRETCH_ENABLED=0` to skip the gesture;
  `COLLIE_MATCH_STRETCH_SETTLE_S` and `COLLIE_MATCH_REACQUIRE_TIMEOUT_S` control
  the animation wait and fresh-frame reacquisition window.
- The experimental manual pointing panel still exposes the guarded
  `locked_point_v19` runner for later work. It is isolated from the autonomous
  stage sequence; that sequence now stays entirely in Unitree Sport mode for
  the arrival posture and return transition.
- Keeps persistent fruit memory separate from the ephemeral visual track. A
  normal target-loss stop therefore cannot erase what Woof was shown before it
  turned around.
- Supports two explicit return backends. The diagnostic `local_odometry`
  fallback averages fresh `rt/sportmodestate` local `(x, y,
  yaw)` samples and performs the original short-range return. The `nav2`
  backend captures Home from a stable `map -> base_link` sample and delegates
  the complete obstacle-aware return to one `NavigateToPose` goal. Both reject
  unstable Home capture, stale pose, travel outside the configured three-metre
  stage envelope, lack of progress, and timeout. Success always requires a
  measured position error of at most 10 cm and heading error of at most 5
  degrees.
- Runs the initial measured turn through the factory `ObstaclesAvoidClient`
  path that has physically actuated Woof, with forward and lateral motion fixed
  at zero. The controller confirms the avoidance switch, takes remote API
  ownership, waits 0.5 seconds for that handoff to settle, then commands a
  0.80 rad/s yaw. Fresh odometry measures the full turn; the 350 ms heartbeat
  watchdog and independent `StopMove` brake remain active. The unused direct
  `SportClient` path remains available behind `COLLIE_DIRECT_TURN_ENABLED`, but
  it is disabled in the stage image because live commands were acknowledged
  without producing physical yaw.
- Selects the highest-confidence fresh YOLO detection whose label equals the
  saved class, followed by two spatially stable confirmations before moving
  forward. A different physical prop of the same class is intentionally valid.
  Search rotation brakes on the first accepted class detection so the next
  inference confirms a stationary view instead of rotating the fruit out of frame.
  If the fruit is not immediately visible after the measured turn, search uses
  accumulated wrap-safe yaw deltas for an explicit bounded 360-degree scan. It
  never relies on a wrapped start/end heading comparison to spin until timeout.
- Runs `TURNING -> SEARCHING -> CONFIRMING -> APPROACHING -> RETURNING_HOME`
  inside the robot
  service. UI refresh timing cannot interrupt command renewal, and leaving the
  page sends the same emergency stop used by the manual follower.
- Treats disappearance as success only after the selected fruit reached the
  lower camera region. Earlier loss of the saved class is an abort with zero
  commanded velocity. The stage profile currently approaches at 0.30 m/s for
  at most 8 seconds and tolerates three consecutive detector misses; these
  limits preserve the original 2.4 m reach while reducing distance travelled
  between detector updates.
- For `nav2`, the companion service prefers `/hesai/points`, converts the fresh
  cloud to `/scan`, uses RTAB-Map for `map -> odom`, and supplies live local and
  global obstacle costmaps to Nav2. Nav2 commands are relayed back through
  `collie-demo`; no second process publishes Unitree motion. The runtime uses
  one exclusive forward/yaw Sport lease so Nav2 remains the only steering
  planner, clamps reverse/lateral motion to zero, and retains the independent
  350 ms `StopMove` watchdog. Home capture and motion remain locked until the
  physical `base_link -> hesai_lidar` transform, odometry, scan, map,
  localization, and `NavigateToPose` server are all fresh and validated.
- The map backend aborts if localization, the map, or the Hesai-derived scan
  becomes stale; if Nav2 rejects or aborts the path; if no velocity heartbeat
  arrives; if less than 4 cm of progress occurs in six seconds; or if final
  measured error exceeds 10 cm or 5 degrees. This layering follows
  [`autonomous-go2-inspection`](https://github.com/Manas-arumalla/autonomous-go2-inspection),
  while keeping Home captured per round and preserving `collie-demo` as the
  only hardware motion owner.

Every class emitted by the local model is selectable from the detection list.
Whale color detection and whale motion targets have been removed.

## Live pointing policy

Open the main UI and use the `Experimental pointing policy (manual only)` panel:

1. Click `Manual Select` on the fruit whose bounding box should drive the paw.
2. Clear the robot and target area, then click `Prepare Standing Point`.
3. Wait for the camera and selected box to settle. The Run button remains
   disabled until the target is stable, freshly YOLO-verified, and above its
   configured class threshold.
4. Click `Run 6.0s Standing Point`. The endpoint returns immediately while the
   robot process runs the 50 Hz actor and 500 Hz low-level publisher.
5. Use either `Stop & Restore Sport Mode` or the global `Stop Now`. Stop sends
   an interrupt to the policy runner and waits for its guarded joint return and
   Sport-controller restoration; it never force-kills the motor owner.

This integration is locally validated but not yet hardware-qualified. The
upstream `locked_point_v19` handoff report records a clean low-level lift but a
guarded abort after 34 policy ticks when estimated torque reached 22.190 Nm
against the 22 Nm ceiling. Do not describe the point as stage-ready until a
supervised run completes the full gesture and recovery without weakening that
guard.

The policy endpoints are:

```text
POST /api/pointing/prepare
POST /api/pointing/run
POST /api/pointing/stop
```

`/api/status` reports the policy phase, readiness, target label, guard limits,
last safety report, peak roll/speed/torque, confidence range, policy ticks, and
whether Sport mode was restored. The browser supplies explicit confirmation
phrases for the two physical steps; direct API callers must supply the same
phrases.

## Model

Download the official YOLOE base checkpoint, capture a clean reference frame,
and bake the three stage props into a self-contained local checkpoint:

```sh
mkdir -p models/candidates models/collie
curl -L --fail \
  "https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-11m-seg.pt" \
  -o models/candidates/yoloe-11m-seg.pt

python tools/build_visual_prompt_weights.py \
  --weights models/candidates/yoloe-11m-seg.pt \
  --reference captures/three-fruit-benchmark/raw/frame_000.jpg \
  --output models/collie/collie-fruit-yoloe11m.pt \
  --device mps \
  --prompt apple=1542,758,1596,817 \
  --prompt banana=1186,779,1262,844 \
  --prompt pear=812,775,884,855
```

The current baked detector checkpoint is 59,997,395 bytes with SHA-256
`7c75fcc5d449a8b00785dfd0c955cbf11bd6bde6a5ede1ea8d34c097413bc53e`.
Detector model files and camera captures are excluded from Git, but the Docker
build context includes the baked detector checkpoint. The 443 KiB standing
actor is intentionally committed at
`models/pointing/locked_point_actor.jit`; startup rejects it unless its SHA-256
is `2e4d1bc727370148f34af6f271c2b264209116a7f8ce3cd3970619507317729e`.

## Local test

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[fruit,test]'
pytest
```

Run on a Mac camera after granting camera permission:

```sh
collie-fruit-webcam \
  --model models/collie/collie-fruit-yoloe11m.pt \
  --camera 0 \
  --confidence 0.7
```

Run the browser UI against a JPEG source:

```sh
collie-fruit-ui \
  --model models/collie/collie-fruit-yoloe11m.pt \
  --source http://woof.local:8096/camera.jpg \
  --port 8097
```

## Deploy to Woof

The voice image reads Go2 camera/microphone media through one WebRTC peer; the
main image reads brokered frames locally and retains the sole motion client on
the interface selected by `GO2_NETWORK_INTERFACE`. The UI binds to port 8096:

```sh
wendy --device woof.local run --yes --detach --restart-on-failure
```

Verify the actual deployed runtime before considering it ready:

```sh
wendy --device woof.local device ps --json
curl http://woof.local:8096/api/status
curl http://woof.local:8098/api/status
curl http://woof.local:8098/api/camera/status
```

Then open `http://woof.local:8096/`. A healthy status response must report the
Collie YOLOE model path, `produce.class_thresholds` of `apple: 0.7`,
`banana: 0.2`, and `pear: 0.7`, the selected fruit and its current observation,
`produce.device.resolved: "cuda:0"`, motion state, and `ok: true`. In this
stage configuration the model only returns apple, banana, and pear detections.

The raw, unannotated camera frame is available at `/camera-raw.jpg` for
repeatable detector evaluation. The fixed-scene benchmark used to tune the
three thresholds can be rerun with:

```sh
python tools/benchmark_fruit_models.py \
  models/collie/collie-fruit-yoloe11m.pt \
  --source captures/three-fruit-benchmark/raw \
  --device mps \
  --output captures/three-fruit-benchmark/results/collie-yoloe.json
```

The stage UI uses `/camera-stream.mjpg`, a persistent stream that forwards the
JPEG already encoded once by the voice WebRTC broker. The main runtime requests
the latest fresh frame from `GET /api/camera/frame.jpg`; broker transport health
is exposed at `GET /api/camera/status` and inside the voice `/api/status`.
`/api/status` on the main runtime reports `camera_fps`, dimensions, frame age,
the broker generation, and independent YOLO inference time. Capture consumption
is capped at 10 Hz (`COLLIE_CAMERA_HZ`), and frames at least 750 ms old are never
served or accepted. Legacy annotated snapshots are limited to 5 Hz
(`COLLIE_ANNOTATED_HZ`) so display fluidity is not gated by inference. The
`camera_rpc` block keeps persistent broker request, success, error, source-age,
and generation telemetry. Set `COLLIE_CAMERA_SOURCE=unitree_rpc` only for the
explicit legacy diagnostic path; it is not the stage default.

The single-peer broker is locally test-validated but is not yet physically
qualified on Woof. Treat the camera issue as open until a live 30-minute soak
and forced peer-loss/recovery test show fresh video while mic and bark remain
usable; mission actuation must be tested separately under operator supervision.

Run the read-only camera and detector soak from a machine that can reach Woof:

```sh
python tools/camera_fruit_soak.py \
  --duration-s 1800 \
  --output artifacts/soak/camera-30m.jsonl
```

The probe polls the main and voice status endpoints without selecting a target
or sending a motion command. It fails on stale camera/detector ages, mismatched
stream generations, frozen broker/runtime/detector frame counters, unhealthy
camera or GPU state, request errors, or an unexpected reconnect. Every sample
and a final summary are written as JSON Lines for later diagnosis.

For a recognition trial, hold exactly one requested stage fruit in view and
require it to appear in at least 80% of fresh samples after a short warm-up:

```sh
python tools/camera_fruit_soak.py \
  --duration-s 60 \
  --expected-fruit pear \
  --minimum-recognition-ratio 0.80 \
  --output artifacts/soak/pear-01.jsonl
```

Repeat the recognition trial ten times for each of `apple`, `banana`, and
`pear`. Exit status `0` is a pass, `1` is a reliability failure, and `2` is an
invalid probe configuration.

The stage control sequence is: click `Select` beside the desired detection, wait for
the Follow button to enable, then click `Follow Selected Fruit` once. `STOP
NOW` remains available throughout motion. Do not run the demo unless the header
shows `STAGE READY`.

## Remember-and-find stage sequence

1. Hold one detected fruit steady and press `Save Class` on that detection.
2. Confirm the saved label and the `MEMORIZED` mission state.
3. Place one fruit of that class in the search area behind Woof. Clear the turn,
   approach, and return paths.
4. Press `Run Remember & Find` once. Woof performs a heading-measured turn,
   requires a multi-frame class lock, stretches, and stops at
   `WAITING FOR GO`.
5. Confirm the target and path are still clear, then press `Go to Fruit`. Woof
   revalidates the saved class in fresh frames, hands it to the guarded follower,
   and returns to the start pose using the configured return backend.
6. Use `STOP NOW` at any time. `Reset Round` erases the saved class.

The mission endpoints are `POST /api/memory/capture`, `DELETE /api/memory`,
`GET /api/memory/reference.jpg`, `POST /api/demo/start`, and
`POST /api/demo/go`, and `POST /api/demo/stop`. `/api/status` reports `memory`,
`mission`, heading age, class-lock state, turn progress, and the terminal reason.

The stage image defaults to `COLLIE_RETURN_BACKEND=nav2`. At mission start,
`collie-demo` asks the separate `collie-nav2` service to capture a stable
`map`-frame Home pose. After the arrival posture completes, the runtime submits
that saved pose to Nav2 and does not report mission success until both the
position and heading tolerances are independently verified. If localization,
the map, the Hesai scan, Nav2 lifecycle state, or the motion handoff is
unhealthy, the sequence aborts and remains stopped. Set
`COLLIE_RETURN_BACKEND=local_odometry` only as an explicit diagnostic fallback;
it does not provide global obstacle-detour planning.

The feature is controlled by `COLLIE_MEMORY_DEMO_ENABLED` and
`COLLIE_AUTONOMOUS_TURN_ENABLED`. Matching, turn speed/angle, search limits,
and arrival geometry are environment-configurable in the Dockerfile. Disabling
either feature does not remove or weaken the manual follower and STOP path.

## Voice stage sequence

1. Verify the page reports `LISTENING`, a fresh microphone age, Scribe
   connected, and the main header reports `STAGE READY`.
2. Clear the full turn, approach, and return paths.
3. Say exactly `apple`, `banana`, or `pear`, or type one of those labels into
   the voice panel and press `Run [fruit] Full Sequence`.
4. Woof barks, captures Home from a stable localization window, turns,
   searches for that YOLO class without running Hello or Stretch, automatically
   revalidates and approaches it, lies down for five seconds, stands up, and
   returns to Home's saved position and heading. When `nav2` is enabled, that
   return is a planned map-frame path around live costmap obstacles.
5. After return-home completes, the voice status returns to `LISTENING` and a
   new fruit word can start the next round.
6. Say `Stop`, `Abort mission`, or press `STOP NOW` to invoke the same emergency
   stop boundary.

The browser exposes a typed fruit-command field with one explicit full-sequence
button plus voice Start, Stop, and Test Bark controls. No additional Go input is
needed after that button is pressed: the existing voice mission releases its
internal Go only after a fresh class lock and remains active through the guarded
return Home. Typed labels enter through `POST /api/command` on the voice service
and use the same bark, preflight, and guarded-mission path as committed speech.
The voice service owns no motion client: it can only call
`POST /api/voice/mission` with the exact `VOICE COMMAND HEARD` confirmation.
The Collie runtime still owns freshness checks, class locking, velocity leases,
watchdogs, arrival classification, and return-home. Oliver's-desk Thor captures
the USB speakerphone through WendyOS audio when that stream is live; the Go2
microphone remains connected as an automatic fallback. The Thor speaker
handles spoken stage confirmations in either case.
