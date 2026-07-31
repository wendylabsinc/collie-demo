"""Out-of-process health watchdog with an independent Unitree StopMove path."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from typing import Mapping
import urllib.error
import urllib.request


def process_probe_ok(status_code: int, payload: object) -> bool:
    """Return true when the child API is alive, even if sensors are degraded.

    ``stage_ready`` is an operator/motion gate, not a process-health signal.
    Restarting the process whenever a GPU inference briefly ages out destroys
    the in-memory fruit reference and can interrupt a measured turn. The child
    runtime already emergency-stops on stale camera/target data; this outer
    watchdog is only responsible for a dead or unreachable child process.
    """

    return (
        status_code == 200
        and isinstance(payload, dict)
        and payload.get("ok") is True
    )


def process_failure_is_fatal(
    *,
    failures: int,
    failure_limit: int,
    elapsed_s: float,
    startup_grace_s: float,
) -> bool:
    """Delay unreachable-process enforcement during CUDA/model warm-up.

    The child can serve camera health before its first CUDA inference. That
    cold inference can temporarily occupy the process for several seconds on
    a Jetson. Motion is still disarmed during this grace period, and a child
    that actually exits is handled immediately by the parent loop.
    """

    return elapsed_s >= startup_grace_s and failures >= failure_limit


def emergency_brake() -> None:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.go2.sport.sport_client import SportClient

    interface = os.environ.get("GO2_NETWORK_INTERFACE", "").strip()
    ChannelFactoryInitialize(0, interface) if interface else ChannelFactoryInitialize(0)
    client = SportClient()
    client.SetTimeout(2.0)
    client.Init()
    result = client.StopMove()
    if result != 0:
        raise RuntimeError(f"StopMove returned {result!r}")


def pointing_warmup_command(
    environ: Mapping[str, str] | None = None,
) -> list[str] | None:
    """Build a CPU-only policy warm-up command when pointing is enabled."""

    values = os.environ if environ is None else environ
    enabled = values.get("COLLIE_POINTING_ENABLED", "").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return None
    policy_path = values.get("COLLIE_POINTING_POLICY", "").strip()
    if not policy_path:
        raise RuntimeError(
            "COLLIE_POINTING_POLICY is required when pointing is enabled"
        )
    script = (
        "import sys, torch; "
        "from collie_demo.pointing_contract import LOCKED_POINT_OBSERVATION_SIZE; "
        "model=torch.jit.load(sys.argv[1], map_location='cpu').eval(); "
        "sample=torch.zeros((1, LOCKED_POINT_OBSERVATION_SIZE), dtype=torch.float32); "
        "model(sample); "
        "print('collie supervisor: pointing policy warm-up complete', flush=True)"
    )
    return [sys.executable, "-c", script, policy_path]


def warm_pointing_policy() -> None:
    """Populate the child-process page cache before the API becomes ready."""

    command = pointing_warmup_command()
    if command is None:
        return
    subprocess.run(
        command,
        check=True,
        timeout=float(os.environ.get("COLLIE_POINTING_WARMUP_TIMEOUT_S", "120")),
    )


def main() -> None:
    port = int(os.environ.get("COLLIE_PORT", "8096"))
    url = f"http://127.0.0.1:{port}/api/status"
    warm_pointing_policy()
    child = subprocess.Popen([sys.executable, "-m", "collie_demo.main"])
    terminating = False

    def terminate(_signum: int, _frame: object) -> None:
        nonlocal terminating
        terminating = True
        try:
            emergency_brake()
        except Exception as exc:
            print(f"emergency brake during shutdown failed: {exc}", file=sys.stderr)
        if child.poll() is None:
            child.terminate()

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    started_at = time.monotonic()
    deadline = started_at + 75.0
    startup_grace_s = max(
        0.0,
        float(os.environ.get("COLLIE_SUPERVISOR_STARTUP_GRACE_S", "60")),
    )
    failure_limit = int(os.environ.get("COLLIE_SUPERVISOR_FAILURE_LIMIT", "80"))
    healthy_once = False
    failures = 0
    while child.poll() is None and not terminating:
        try:
            with urllib.request.urlopen(url, timeout=0.5) as response:
                payload = json.load(response)
                probe_ok = process_probe_ok(response.status, payload)
        except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
            probe_ok = False
        if probe_ok:
            healthy_once = True
            failures = 0
        elif healthy_once:
            failures += 1
        elif time.monotonic() >= deadline:
            failures = failure_limit
        if process_failure_is_fatal(
            failures=failures,
            failure_limit=failure_limit,
            elapsed_s=time.monotonic() - started_at,
            startup_grace_s=startup_grace_s,
        ):
            print(
                "collie supervisor: child API unreachable beyond tolerance "
                f"(failures={failures}, limit={failure_limit}, "
                f"elapsed_s={time.monotonic() - started_at:.1f})",
                file=sys.stderr,
                flush=True,
            )
            try:
                emergency_brake()
            finally:
                child.terminate()
                try:
                    child.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            raise SystemExit(1)
        time.sleep(0.25)
    if child.poll() is None:
        child.wait()
    raise SystemExit(child.returncode or 0)


if __name__ == "__main__":
    main()
