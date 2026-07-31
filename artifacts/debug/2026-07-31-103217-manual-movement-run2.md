# Woof manual-movement diagnostic, run 2

- Window: 2026-07-31 10:32:17-10:32:27 -0700
- Duration: 10 seconds
- Motion authority: disarmed throughout; Woof was moved manually
- Camera source: `voice_webrtc_camera_broker`
- Camera generation: `2333db7ef5244dd58e2d5a22354ca984.1`

## Corrected capture configuration

- HTTP request deadline increased from 0.35 to 1.5 seconds.
- Response latency recorded for every main and voice request.
- Onboard sampler changed to a POSIX-compatible counter and validated before
  movement. It records load, memory, temperature, app CPU/memory, GPU load, and
  wired/Wi-Fi counters.

## During movement

- Main status sample 0 exceeded 1.5 seconds.
- Main status sample 1 succeeded in 0.054 seconds at 10:32:18.738.
- Main status samples 2-6 all exceeded 1.5 seconds.
- All six voice status requests exceeded 1.5 seconds.
- The already-validated Wendy onboard attachment could not establish during the
  window, so no onboard samples were returned while Woof was moving.
- At the one successful main sample:
  - Stage ready: true
  - Main camera: 9.4 FPS, 0.048 s frame age
  - WebRTC source age: 0.007 s
  - Broker errors: 39, consecutive errors: 0
  - TensorRT inference: 107.5 ms on CUDA
  - Motion: disarmed, zero command, no fault

## Immediately after movement

- Main camera: 10.0 FPS, 0.108 s frame age
- WebRTC source: 14.2 FPS at 1280x720, 0.066 s frame age
- Broker errors: still 39; consecutive errors: 0
- WebRTC camera reconnects: 0; stale reconnects: 0
- TensorRT inference: 145.3 ms on CUDA
- Scribe state: reconnecting after `Connection to remote host was lost.`
- Host load: 10.55 / 9.63 / 8.59
- Available memory: 10,869.6 MB
- Maximum temperature: 58.6 C
- GPU load: 468 permille (46.8 percent)
- Main app CPU/memory: 55.0 percent / 8.8 percent
- Internal Ethernet: zero RX/TX errors or drops
- Motion/navigation: disarmed, zero command, no fault

## Interpretation

The movement did not break the WebRTC camera or TensorRT pipeline. The camera
generation remained unchanged, broker errors did not increase, and there were
no camera reconnects. The reproduced failure is host/API reachability or
scheduling latency: both HTTP services and the Wendy attachment became
unresponsive during movement, while the internal camera stream stayed healthy.
The simultaneous Scribe network disconnect and elevated host load narrow the
remaining candidates to external network instability and CPU scheduling pressure;
they do not support a camera-model failure.
