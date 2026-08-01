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
  when either the Jetson or Go2 IMU temperature crosses the configured warning
  boundary; and
- asks that same audio peer to announce “Low battery” once when state of charge
  reaches 25%, rearming after charge recovers to 30%. Failed audio delivery is
  retried every 60 seconds while the battery remains low.

The app never publishes a Unitree command and does not stop, move, throttle, or
shut down the robot.

Both audible alerts require Collie's voice service on port 8098. Their live API
status records whether each audio request succeeded instead of silently
claiming an announcement played.

## Thresholds

The default monitored thresholds are:

- warning: 85 C sustained for 10 seconds;
- critical: 95 C immediately;
- clear: below 80 C for 120 seconds; and
- repeat the beep every 60 seconds while an alert remains active.

Woof's Jetson reports a 99 C software-throttling trip and a 104.5 C critical
shutdown trip. The same configurable thresholds are also used as a conservative
operational guard for the Go2 IMU; they are not presented as Unitree factory
limits. Go2 motor, battery, and NTC temperatures are recorded but do not trigger
this alarm until their component-specific limits are qualified.

## Deploy

First deploy the root `collie-demo` application so its voice service provides
`POST http://127.0.0.1:8098/api/thermal/beep`. Then deploy this directory:

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
