"""Fail-closed HTTP client for the separate ROS 2/Nav2 return service.

The Nav2 service owns map-frame localization and path planning. It never talks
to Unitree hardware directly: its velocity relay uses collie-demo's existing
navigation endpoints, so collie-demo's exclusive direct-Sport lease and
watchdog remain the single hardware authority.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import math
import time
from typing import Any, Protocol
from urllib import error as urlerror
from urllib import request as urlrequest


TERMINAL_NAVIGATION_STATES = frozenset(
    {"succeeded", "failed", "cancelled", "aborted"}
)


class Nav2ReturnError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class MapHomeCapture:
    x_m: float
    y_m: float
    yaw_rad: float
    frame_id: str
    sample_count: int
    maximum_position_span_m: float
    maximum_yaw_span_rad: float

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "MapHomeCapture":
        home = payload.get("home")
        validation = payload.get("validation")
        if not isinstance(home, dict) or not isinstance(validation, dict):
            raise Nav2ReturnError("Nav2 Home response is incomplete")
        try:
            capture = cls(
                x_m=float(home["x_m"]),
                y_m=float(home["y_m"]),
                yaw_rad=float(home["yaw_rad"]),
                frame_id=str(home["frame_id"]),
                sample_count=int(validation["sample_count"]),
                maximum_position_span_m=float(
                    validation["maximum_position_span_m"]
                ),
                maximum_yaw_span_rad=math.radians(
                    float(validation["maximum_yaw_span_deg"])
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise Nav2ReturnError("Nav2 Home response is invalid") from exc
        if (
            capture.frame_id != "map"
            or capture.sample_count < 2
            or not all(
                math.isfinite(value)
                for value in (
                    capture.x_m,
                    capture.y_m,
                    capture.yaw_rad,
                    capture.maximum_position_span_m,
                    capture.maximum_yaw_span_rad,
                )
            )
        ):
            raise Nav2ReturnError("Nav2 Home response failed validation")
        return capture

    def pose_dict(self) -> dict[str, float | str]:
        return {
            "x_m": round(self.x_m, 4),
            "y_m": round(self.y_m, 4),
            "yaw_rad": round(self.yaw_rad, 4),
            "frame_id": self.frame_id,
        }

    def validation_dict(self) -> dict[str, float | int | str]:
        return {
            "source": "nav2_map_localization",
            "sample_count": self.sample_count,
            "maximum_position_span_m": round(
                self.maximum_position_span_m, 4
            ),
            "maximum_yaw_span_deg": round(
                math.degrees(self.maximum_yaw_span_rad), 2
            ),
        }


@dataclass(frozen=True, slots=True)
class Nav2ReturnStatus:
    state: str
    reason: str
    distance_remaining_m: float | None
    position_error_m: float | None
    heading_error_rad: float | None
    navigation_time_s: float | None
    recoveries: int
    localization_healthy: bool
    map_healthy: bool
    scan_healthy: bool
    motion_armed: bool

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "Nav2ReturnStatus":
        navigation = payload.get("navigation")
        health = payload.get("health")
        motion = payload.get("motion")
        if not isinstance(navigation, dict):
            raise Nav2ReturnError("Nav2 status is missing navigation state")
        if not isinstance(health, dict):
            health = {}
        if not isinstance(motion, dict):
            motion = {}

        def optional_float(name: str) -> float | None:
            value = navigation.get(name)
            if value is None:
                return None
            rendered = float(value)
            if not math.isfinite(rendered):
                raise Nav2ReturnError(f"Nav2 status {name} is non-finite")
            return rendered

        state = str(navigation.get("state") or "unavailable").lower()
        heading_error = navigation.get("heading_error_deg")
        return cls(
            state=state,
            reason=str(navigation.get("reason") or state),
            distance_remaining_m=optional_float("distance_remaining_m"),
            position_error_m=optional_float("position_error_m"),
            heading_error_rad=(
                None
                if heading_error is None
                else math.radians(float(heading_error))
            ),
            navigation_time_s=optional_float("navigation_time_s"),
            recoveries=int(navigation.get("recoveries") or 0),
            localization_healthy=bool(health.get("localization_healthy")),
            map_healthy=bool(health.get("map_healthy")),
            scan_healthy=bool(health.get("scan_healthy")),
            motion_armed=bool(motion.get("armed")),
        )

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_NAVIGATION_STATES

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "reason": self.reason,
            "distance_remaining_m": self.distance_remaining_m,
            "position_error_m": self.position_error_m,
            "heading_error_deg": None
            if self.heading_error_rad is None
            else round(math.degrees(self.heading_error_rad), 2),
            "navigation_time_s": self.navigation_time_s,
            "recoveries": self.recoveries,
            "localization_healthy": self.localization_healthy,
            "map_healthy": self.map_healthy,
            "scan_healthy": self.scan_healthy,
            "motion_armed": self.motion_armed,
        }


class Nav2ReturnClientProtocol(Protocol):
    async def health(self) -> dict[str, Any]: ...

    async def capture_home(
        self,
        *,
        duration_s: float,
        maximum_position_span_m: float,
        maximum_yaw_span_rad: float,
    ) -> MapHomeCapture: ...

    async def start_return(
        self,
        *,
        position_tolerance_m: float,
        heading_tolerance_rad: float,
        timeout_s: float,
    ) -> Nav2ReturnStatus: ...

    async def status(self) -> Nav2ReturnStatus: ...

    async def cancel(self, reason: str) -> None: ...


@dataclass(frozen=True, slots=True)
class Nav2ReturnClientConfig:
    base_url: str = "http://127.0.0.1:8100"
    request_timeout_s: float = 2.0

    def __post_init__(self) -> None:
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("Nav2 base URL must be HTTP or HTTPS")
        if (
            not math.isfinite(self.request_timeout_s)
            or self.request_timeout_s <= 0.0
        ):
            raise ValueError("Nav2 request timeout must be positive")


class Nav2ReturnClient:
    def __init__(self, config: Nav2ReturnClientConfig) -> None:
        self.config = config

    async def health(self) -> dict[str, Any]:
        return await self._request("GET", "/api/health")

    async def capture_home(
        self,
        *,
        duration_s: float,
        maximum_position_span_m: float,
        maximum_yaw_span_rad: float,
    ) -> MapHomeCapture:
        payload = await self._request(
            "POST",
            "/api/home/capture",
            {
                "duration_s": duration_s,
                "maximum_position_span_m": maximum_position_span_m,
                "maximum_yaw_span_deg": math.degrees(maximum_yaw_span_rad),
            },
            timeout_s=max(self.config.request_timeout_s, duration_s + 2.0),
        )
        return MapHomeCapture.from_payload(payload)

    async def start_return(
        self,
        *,
        position_tolerance_m: float,
        heading_tolerance_rad: float,
        timeout_s: float,
    ) -> Nav2ReturnStatus:
        payload = await self._request(
            "POST",
            "/api/home/navigate",
            {
                "position_tolerance_m": position_tolerance_m,
                "heading_tolerance_deg": math.degrees(heading_tolerance_rad),
                "timeout_s": timeout_s,
            },
        )
        return Nav2ReturnStatus.from_payload(payload)

    async def status(self) -> Nav2ReturnStatus:
        payload = await self._request("GET", "/api/status")
        return Nav2ReturnStatus.from_payload(payload)

    async def cancel(self, reason: str) -> None:
        try:
            await self._request(
                "POST",
                "/api/navigation/cancel",
                {"reason": str(reason)[:240]},
            )
        except Nav2ReturnError:
            # The independent collie-demo emergency stop remains authoritative.
            return

    async def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._request_sync,
            method,
            path,
            payload,
            timeout_s,
        )

    def _request_sync(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None,
        timeout_s: float | None,
    ) -> dict[str, Any]:
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urlrequest.Request(
            f"{self.config.base_url.rstrip('/')}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        started = time.monotonic()
        try:
            with urlrequest.urlopen(
                request,
                timeout=timeout_s or self.config.request_timeout_s,
            ) as response:
                raw = response.read()
        except urlerror.HTTPError as exc:
            try:
                body = json.loads(exc.read().decode("utf-8"))
                detail = body.get("detail") or body.get("reason")
            except Exception:
                detail = None
            raise Nav2ReturnError(
                str(detail or f"Nav2 gateway returned HTTP {exc.code}")
            ) from exc
        except (OSError, TimeoutError, urlerror.URLError) as exc:
            raise Nav2ReturnError(
                "Nav2 gateway is unavailable "
                f"after {time.monotonic() - started:.2f}s: {exc}"
            ) from exc
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Nav2ReturnError("Nav2 gateway returned invalid JSON") from exc
        if not isinstance(parsed, dict):
            raise Nav2ReturnError("Nav2 gateway returned a non-object response")
        if parsed.get("ok") is False:
            raise Nav2ReturnError(
                str(parsed.get("detail") or parsed.get("reason") or "Nav2 failed")
            )
        return parsed
