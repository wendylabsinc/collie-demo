# Woof thermal monitor

`woof-thermal-monitor` is an independent, read-only Wendy app for the Go2. It:

- samples every Jetson thermal zone every two seconds;
- subscribes read-only to `rt/lowstate` for the IMU temperature, all 12 motor
  temperatures, battery thermistors, internal NTC sensors, fan telemetry, and
  power;
- writes a min/average/max aggregate to persistent SQLite storage every 30
  seconds;
- keeps 30 days of history; and
- asks Collie's existing single WebRTC AudioHub peer to play three short beeps
  when the Jetson, Go2 IMU, or an individual motor crosses its configured
  warning boundary, or when a motor heats unusually quickly; and
- asks that same audio peer to announce “Low battery” once when state of charge
  reaches 25%, rearming after charge recovers to 30%. Failed audio delivery is
  retried every 60 seconds while the battery remains low.

The app never publishes a Unitree command and does not stop, move, throttle, or
shut down the robot.

Go2 DDS initialization is retried in the background every two seconds when the
robot-side Ethernet interface is not ready yet. Jetson sampling and the HTTP API
remain available during recovery. Configure the interval with
`WOOF_GO2_RECONNECT_S` (seconds, default `2`, valid range `0.1..60`); this changes
only read-only telemetry recovery and grants no motion authority.

The monitor first reuses Collie's voice service on port 8098 when it is running.
If it is stopped, the monitor opens a short-lived, audio-only WebRTC connection
to the Go2, plays the alarm through the robot speaker, and disconnects. It does
not enable the camera, microphone, speech recognition, or motion stack. Live API
status records whether each audio request actually succeeded. The direct path
uses bounded backoff for the Go2's WebRTC handshake rate limit.

## Thresholds

The default monitored thresholds are:

- warning: 85 C sustained for 10 seconds;
- critical: 95 C immediately;
- clear: below 80 C for 120 seconds; and
- repeat the beep every 60 seconds while an alert remains active.

Individual motors use a separate, more conservative policy:

- warning: 70 C sustained for 10 seconds;
- critical: 80 C immediately;
- rapid rise: at least 8 C at 5 C/min or faster over a rolling three-minute
  window, sustained for 10 seconds; and
- clear: below 65 C with a low rise rate for 120 seconds.

Woof's Jetson reports a 99 C software-throttling trip and a 104.5 C critical
shutdown trip. The same configurable thresholds are also used as a conservative
operational guard for the Go2 IMU; they are not presented as Unitree factory
limits. The motor thresholds are similarly operational safeguards chosen from
Woof's observed behavior, not claimed Unitree factory shutdown limits. Battery
and NTC temperatures remain recorded without driving the alarm.

## Deploy

Deploy this directory directly; the root `collie-demo` application is optional:

```sh
cd thermal_monitor
wendy --device woof.local run --yes --detach --restart-on-failure
```

Status and recent history:

```sh
curl http://woof.local:8102/api/status
curl 'http://woof.local:8102/api/history?limit=120'
```

The persistent database is `/state/thermal.sqlite3` inside the app's
`woof-thermal-monitor-state` volume.
