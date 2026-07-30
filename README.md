# Collie Demo

A self-contained Wendy app for Woof that can save a locally detected fruit class,
turn around, recognize a fresh fruit of that class, safely approach it, and return
to its saved start pose. The
validated manual apple/banana/pear follower remains available as a fallback. It uses a
YOLOE-11m checkpoint whose visual prompt embeddings were baked from the three
physical stage props. The Woof image runs a TensorRT 10.7 FP16 engine exported
on its Orin; the original PyTorch checkpoint remains in the image as the source
artifact.
Frames come directly from Unitree's public `VideoClient`; inference,
annotation, motion supervision, and the browser UI all run locally in the
robot container. No Roboflow service, hosted inference API, Hugging Face key,
or internet connection is used for vision inference. The optional voice service
streams microphone PCM to ElevenLabs Scribe v2 Realtime; motion control and
fruit inference remain on Woof.

## Current behavior

- Bundles the local PyTorch source checkpoint and Woof-specific TensorRT FP16
  engine; runtime inference does not call a hosted API.
- Detects only the three visual-prompt classes: apple, banana, and pear.
- Bundles the hash-pinned iteration-42,500 TorchScript pointing actor and
  exposes a guarded browser flow: manually select a fresh fruit box, request
  Unitree `StandDown`, then run one second of the real full-gain policy. The
  actor receives the exact normalized selected `xyxy` box plus live Go2
  proprioception; it cannot silently switch to another detection.
- Keeps the pointing runner behind the same single-motion-owner boundary as
  following and navigation. The UI exposes Stop, target changes are rejected
  during low-level control, target loss aborts, and every exit path attempts to
  return to the captured StandDown pose and restore Sport mode.
- Retains the 0.30 rad roll, 0.35 rad pitch, 12-unit estimated-torque,
  2 rad/s joint-speed, 0.60 rad/s target-rate, low-state freshness, policy
  checksum, unchanged target-lock, and 800 ms bbox-age guards. The stage runner
  contains no roll-guard bypass. The visible full-gain segment is capped at one
  second because the earlier three-second hardware test reached the roll guard
  after 67 policy ticks.
- Runs a separate persistent Go2 WebRTC microphone service on port 8098. It
  accepts only deterministic bare `apple|banana|pear` commands, the legacy
  `Find [the] ...` form, plus `stop|abort|cancel`; arbitrary transcripts can
  never become motor commands.
- A committed bare `apple`, `banana`, or `pear` command (with `pair` accepted
  as a Scribe homophone for pear) plays the user-supplied native AudioHub bark,
  stores only the requested YOLO class, and starts the normal guarded mission.
  The voice-only path skips the initial Hello and recognition Stretch gestures.
  The voice bridge automatically releases Go only after the mission reports a
  fresh multi-frame class lock and all stage-health checks remain ready.
- When the arrival lay-down enters its five-second hold, the voice bridge
  plays exactly one AudioHub bark. It keeps later fruit commands gated until the
  return-home controller reports completion or the mission safely aborts.
- Once repeated detections confirm a fruit at the lower camera edge, a final
  10 cm nudge that factory avoidance blocks stops cleanly and continues into the
  same StandDown sequence, even when local odometry reports no additional
  distance. Camera/pose staleness and target losses before lower-edge
  confirmation still abort.
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
- After the fruit has reached the lower camera region and then disappears, Woof
  first releases every locomotion owner and stops. The stage image calls the
  stock `StandDown`, holds the completed lay-down posture for five seconds,
  calls the paired Unitree `StandUp`, and only then starts the odometry-based
  return home.
  Operator Stop cancels the mission and prevents the return leg from arming.
  Configure the sequence with `COLLIE_ARRIVAL_REST_ENABLED` and
  `COLLIE_ARRIVAL_REST_DURATION_S`. The older stock `Hello` arrival
  acknowledgement remains available behind `COLLIE_ARRIVAL_HELLO_ENABLED`.
- For the configured arrival-pointing class, three distinct detector frames in
  the near region can trigger a different
  handoff before the box disappears: Woof stops, enters `StandDown`, runs the
  hash-pinned one-second bounding-box policy, and verifies that Sport mode was
  restored. If return-home is enabled it then calls `StandUp` before
  reacquiring the factory obstacle-avoidance lease. A policy guard, stale box,
  timeout, or failed controller restoration aborts the mission. Telemetry calls
  this a reach attempt and reports contact as `unverified`; there is no
  independent paw-contact sensor. Configure it with
  `COLLIE_ARRIVAL_POINTING_ENABLED`, `COLLIE_ARRIVAL_POINTING_LABEL`, and
  `COLLIE_ARRIVAL_POINTING_TIMEOUT_S`. It is disabled in the stage image; the
  manual pointing panel remains available for supervised engineering tests.
- Keeps persistent fruit memory separate from the ephemeral visual track. A
  normal target-loss stop therefore cannot erase what Woof was shown before it
  turned around.
- Uses fresh `rt/sportmodestate` IMU yaw and local `(x, y)` odometry. The mission
  refuses to start or aborts if heading or the position required for return-home
  becomes stale; it never estimates a 180-degree turn or return distance from
  elapsed time alone.
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
- Captures Home from fresh local odometry when the operator starts the mission.
  After the fruit is reached, the return controller first uses the measured
  watchdog-protected yaw-only Sport lease to face Home; translation is
  impossible during this in-place correction. It then reacquires the factory
  obstacle-avoidance channel, drives at up to 0.30 m/s, and uses the same
  yaw-only handoff to restore the original heading after reaching Home. It
  stops within a 10 cm position tolerance and aborts on stale pose, a
  45-second timeout, a stalled heading correction, or less than 4 cm of
  translational progress in six seconds. Those progress limits match the
  slower displacement observed behind the factory avoidance controller while
  retaining a bounded fail-stop. This is a short-range open-stage return
  controller, not a global map planner.

Every class emitted by the local model is selectable from the detection list.
Whale color detection and whale motion targets have been removed.

## Local box-tracker experiment

The runtime contains an opt-in KLT/partial-affine image tracker for filling the
spatial gap between GPU YOLO results. OpenCV executes the optical-flow and
RANSAC work in native code; YOLO remains authoritative for the class and
confidence. Tracker-only frames never manufacture or reuse a confidence score,
and every fresh YOLO result continues to revalidate the selected class.

This experiment is disabled by default and does not change the stage image:

```bash
COLLIE_PRODUCE_TRACKER=off collie-demo
```

To exercise it in a local test runtime:

```bash
COLLIE_MOTION_ENABLED=0 \
COLLIE_PRODUCE_TRACKER=klt_affine \
collie-demo
```

The tracker fails closed on insufficient texture, frame-size changes, too few
forward/backward-consistent features, implausible scale, or an implausible
one-frame jump. Do not enable it on Woof until recorded camera sequences have
been benchmarked and shadow-compared with the TensorRT-only baseline.

### MAX/Mojo shadow postprocessor

An optional MAX Graph backend loads the bundled Mojo custom operation and
shadow-computes affine box transformation, clipping, blending, scale checks,
and center-step checks. The Python reference remains authoritative: a Mojo
mismatch or runtime error is counted in tracker metrics and can never replace
the box used by Collie.

The separate `collie-box-shadow` Wendy app serves a read-only A/B monitor on
port 8106. Its two panes use the same fetched camera image:

- **YOLO / TensorRT** shows the latest detector class, confidence, bounding box,
  and inference latency from the active Collie app.
- **KLT + MAX/Mojo** shows the confidence-neutral optical-flow track and the
  MAX/Mojo box-postprocessing latency. MAX/Mojo does not classify fruit and is
  not presented as a second detector.

The monitor reports box IoU, center displacement, detector-gap coverage, KLT
latency, MAX kernel latency, and the resolved MAX device. It has no Unitree,
DDS, voice, or motion client and reports `control_authority=none_shadow_only`.
The Woof deployment currently pins MAX/Mojo to CPU because MAX 26.4 and the
2026-07-29 nightly both emitted `sm_80` kernels while Woof's Jetson Orin
requires `sm_87`; those artifacts fail at execution with
`CUDA_ERROR_NO_BINARY_FOR_GPU`. YOLO remains TensorRT/CUDA. This makes the UI
valid for output-quality and end-to-end latency comparison, but not a
like-for-like GPU throughput benchmark.

Install Modular only in an isolated development environment:

```bash
uv pip install \
  --python .venv/bin/python \
  modular \
  --index https://whl.modular.com/nightly/simple/ \
  --prerelease allow
```

Run deterministic parity and transfer-overhead measurements:

```bash
.venv/bin/collie-box-postprocess-benchmark \
  --device cpu \
  --iterations 200
```

The runtime flags are intentionally separate and disabled by default:

```bash
COLLIE_MOTION_ENABLED=0 \
COLLIE_PRODUCE_TRACKER=klt_affine \
COLLIE_BOX_POSTPROCESS_SHADOW=max_mojo \
COLLIE_BOX_POSTPROCESS_DEVICE=cpu \
collie-demo
```

`COLLIE_BOX_POSTPROCESS_DEVICE=accelerator` is reserved until a compiled
artifact is inspected for `sm_87` and successfully executed on Woof. No
MAX/Mojo result has control authority.

## Separate MAX vision fork

The `codex/max-vision-fork` worktree contains a separate, read-only Wendy app
named `collie-max-vision`. It consumes the same `/camera-raw.jpg` frame and the
same YOLO/TensorRT detections as the production app, but sends up to three
active fruit tracks through a real MAX Graph and the bundled
`collie_temporal_fusion` Mojo custom operation.

The Mojo operation:

- predicts the next box from bounded per-coordinate velocity,
- confidence-weights fresh YOLO measurements,
- rejects implausible center jumps,
- smooths accepted box measurements,
- decays velocity during short detector gaps, and
- clips every result to the camera frame.

It never creates a class or confidence. Solid boxes in the MAX view are
fresh-YOLO measurements fused by MAX; dashed boxes are bounded predictions and
expire after 450 ms. The app has no Unitree, DDS, voice, or motion client and
reports `control_authority=none_read_only`.

Run the fork locally against Woof:

```bash
COLLIE_MAX_VISION_SOURCE_URL=http://woof.local:8096 \
COLLIE_MAX_VISION_DEVICE=cpu \
COLLIE_MAX_VISION_PORT=8107 \
python -m collie_demo.max_vision_server
```

Deploy it separately:

```bash
wendy --device woof.local run \
  --dockerfile Dockerfile.max \
  --detach \
  --restart-unless-stopped \
  --yes
```

The comparison UI is then available at `http://woof.local:8107/`. Keep this
experimental app stopped during the actual stage run unless its added camera
fetch/decode/encode load has passed a full rehearsal.

## Live pointing policy

Open the main UI and use the `Show the real pointing policy` panel:

1. Click `Manual Select` on the fruit whose bounding box should drive the paw.
2. Clear the robot and target area, then click `Lay Woof Down`.
3. Wait for the camera and selected box to settle. The Run button remains
   disabled until the target is stable, freshly YOLO-verified, and above its
   configured class threshold.
4. Click `Run 1.0s Point`. The endpoint returns immediately while the robot
   process runs the 50 Hz actor and 500 Hz low-level publisher.
5. Use either `Stop & Restore Sport Mode` or the global `Stop Now`. Stop sends
   an interrupt to the policy runner and waits for its guarded joint return and
   Sport-controller restoration; it never force-kills the motor owner.

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

The current baked checkpoint is 59,997,395 bytes with SHA-256
`7c75fcc5d449a8b00785dfd0c955cbf11bd6bde6a5ede1ea8d34c097413bc53e`.
Detector model files and camera captures are excluded from Git, but the Docker
build context includes the baked detector checkpoint. The 454 KiB pointing
actor is intentionally committed at
`models/pointing/policy_actor_42500.jit`; startup rejects it unless its SHA-256
is `5ac866353150b82309a083827aefd2f43e779a5ba67c8d617a5b612b89fe1938`.

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

The on-robot image reads the Unitree camera and motion clients over the network
interface selected by `GO2_NETWORK_INTERFACE`, and binds the UI to port 8096:

```sh
wendy --device woof.local run --yes --detach --restart-on-failure
```

Verify the actual deployed runtime before considering it ready:

```sh
wendy --device woof.local device ps --json
curl http://woof.local:8096/api/status
curl http://woof.local:8098/api/status
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
JPEG already supplied by Unitree instead of opening a new request and
re-encoding every displayed frame. `/api/status` reports `camera_fps`, frame
dimensions, frame age, and the independent YOLO inference time. Camera capture
defaults to 30 Hz (`COLLIE_CAMERA_HZ`) while legacy annotated snapshots are
limited to 5 Hz (`COLLIE_ANNOTATED_HZ`) so display fluidity is not gated by
inference or full-resolution JPEG encoding.

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
   and returns to the start pose using local odometry.
6. Use `STOP NOW` at any time. `Reset Round` erases the saved class.

The mission endpoints are `POST /api/memory/capture`, `DELETE /api/memory`,
`GET /api/memory/reference.jpg`, `POST /api/demo/start`, and
`POST /api/demo/go`, and `POST /api/demo/stop`. `/api/status` reports `memory`,
`mission`, heading age, class-lock state, turn progress, and the terminal reason.

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
4. Woof barks, captures Home, turns, searches for that YOLO class without
   running Hello or Stretch, automatically revalidates and approaches it, lies
   down, barks once, remains down for five seconds, stands, and returns to
   Home's saved position and heading.
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
