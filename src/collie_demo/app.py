from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict

from .runtime import CollieRuntime, RuntimeCommandError


class ArmRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    confirmation: str


class TargetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target: str
    center: tuple[int, int] | None = None


class MemoryCaptureRequest(TargetRequest):
    round_id: str | None = None


class NavigationCommandRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    forward_mps: float
    yaw_rps: float


class ForwardCalibrationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    amount_mps: float
    confirmation: str


class Nav2ForwardCalibrationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    amount_mps: float
    confirmation: str


class VoiceEventRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event: str
    transcript: str = ""
    target: str | None = None
    error: str = ""


class VoiceMissionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target: str
    transcript: str
    confirmation: str


def create_app(runtime: CollieRuntime, web_directory: Path) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await runtime.start()
        try:
            yield
        finally:
            await runtime.close()

    app = FastAPI(title="Collie fruit-follow demo", version="1", lifespan=lifespan)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(web_directory / "index.html")

    @app.get("/debug")
    def debug_page() -> FileResponse:
        return FileResponse(web_directory / "debug.html")

    @app.get("/tests")
    def mini_tests_page() -> FileResponse:
        return FileResponse(web_directory / "tests" / "index.html")

    @app.get("/tests/forward-motion")
    def forward_motion_test_page() -> FileResponse:
        return FileResponse(web_directory / "tests" / "forward-motion.html")

    @app.get("/tests/nav2-forward-motion")
    def nav2_forward_motion_test_page() -> FileResponse:
        return FileResponse(web_directory / "tests" / "nav2-forward-motion.html")

    @app.get("/tests/fruit-detector")
    def fruit_detector_test_page() -> FileResponse:
        return FileResponse(web_directory / "tests" / "fruit-detector.html")

    @app.get("/forward-calibration")
    def forward_calibration_page() -> FileResponse:
        return FileResponse(web_directory / "tests" / "forward-motion.html")

    @app.get("/nav2-forward-calibration")
    def nav2_forward_calibration_page() -> FileResponse:
        return FileResponse(web_directory / "tests" / "nav2-forward-motion.html")

    @app.get("/api/status")
    async def status() -> dict[str, object]:
        return await runtime.status()

    @app.get("/camera.jpg")
    async def camera() -> Response:
        jpeg = await runtime.jpeg()
        if jpeg is None:
            raise HTTPException(status_code=503, detail="no camera frame yet")
        return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/camera-stream.mjpg")
    async def camera_stream() -> StreamingResponse:
        async def frames():
            last_frame_id = 0
            while True:
                packet = await runtime.wait_for_stream_frame(last_frame_id)
                if packet is None:
                    return
                last_frame_id, jpeg = packet
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                    + jpeg
                    + b"\r\n"
                )

        return StreamingResponse(
            frames(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/camera-raw.jpg")
    async def raw_camera() -> Response:
        jpeg = await runtime.raw_jpeg()
        if jpeg is None:
            raise HTTPException(status_code=503, detail="no camera frame yet")
        return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.post("/api/arm")
    async def arm(request: ArmRequest) -> dict[str, object]:
        try:
            return await runtime.arm(request.confirmation)
        except RuntimeCommandError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/follow")
    async def follow(request: ArmRequest) -> dict[str, object]:
        try:
            return await runtime.start_follow(request.confirmation)
        except RuntimeCommandError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/target")
    async def target(request: TargetRequest) -> dict[str, object]:
        try:
            return await runtime.select_target(request.target, request.center)
        except RuntimeCommandError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/memory/capture")
    async def capture_memory(request: MemoryCaptureRequest) -> dict[str, object]:
        try:
            return await runtime.remember_target(
                request.target,
                request.center,
                expected_round_id=request.round_id,
            )
        except RuntimeCommandError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/round/reset")
    async def reset_round() -> dict[str, object]:
        return await runtime.reset_round()

    @app.delete("/api/memory")
    async def clear_memory() -> dict[str, object]:
        return await runtime.clear_memory()

    @app.get("/api/memory/reference.jpg")
    async def memory_reference() -> Response:
        jpeg = await runtime.memory_reference_jpeg()
        if jpeg is None:
            raise HTTPException(status_code=404, detail="no fruit is saved")
        return Response(
            jpeg,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/api/demo/start")
    async def start_demo(request: ArmRequest) -> dict[str, object]:
        try:
            return await runtime.start_demo(request.confirmation)
        except RuntimeCommandError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/demo/stop")
    async def stop_demo() -> dict[str, object]:
        return await runtime.stop_demo()

    @app.post("/api/pointing/prepare")
    async def prepare_pointing(request: ArmRequest) -> dict[str, object]:
        try:
            return await runtime.prepare_pointing(request.confirmation)
        except RuntimeCommandError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/pointing/run")
    async def run_pointing(request: ArmRequest) -> dict[str, object]:
        try:
            return await runtime.start_pointing(request.confirmation)
        except RuntimeCommandError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/pointing/stop")
    async def stop_pointing() -> dict[str, object]:
        return await runtime.stop_pointing()

    @app.post("/api/posture/stand")
    async def restore_standing(request: ArmRequest) -> dict[str, object]:
        try:
            return await runtime.restore_standing(request.confirmation)
        except RuntimeCommandError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/calibration/forward-pulse")
    async def calibrate_forward_deadband(
        request: ForwardCalibrationRequest,
    ) -> dict[str, object]:
        try:
            return await runtime.run_forward_calibration(
                request.amount_mps,
                request.confirmation,
            )
        except RuntimeCommandError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/calibration/nav2-forward-pulse")
    async def calibrate_nav2_forward_deadband(
        request: Nav2ForwardCalibrationRequest,
    ) -> dict[str, object]:
        try:
            return await runtime.run_nav2_forward_calibration(
                request.amount_mps,
                request.confirmation,
            )
        except RuntimeCommandError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/demo/go")
    async def approve_demo_go(request: ArmRequest) -> dict[str, object]:
        try:
            return await runtime.approve_demo_go(request.confirmation)
        except RuntimeCommandError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/voice/event")
    async def voice_event(request: VoiceEventRequest) -> dict[str, object]:
        return await runtime.record_voice_event(
            event=request.event,
            transcript=request.transcript,
            target=request.target,
            error=request.error,
        )

    @app.post("/api/voice/mission")
    async def voice_mission(request: VoiceMissionRequest) -> dict[str, object]:
        try:
            return await runtime.start_voice_mission(
                request.target,
                request.transcript,
                request.confirmation,
            )
        except RuntimeCommandError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/pulse")
    async def pulse() -> dict[str, object]:
        try:
            return await runtime.pulse()
        except RuntimeCommandError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/stop")
    async def stop() -> dict[str, object]:
        return await runtime.stop()

    @app.get("/api/navigation/status")
    async def navigation_status() -> dict[str, object]:
        return await runtime.navigation_status()

    @app.post("/api/navigation/arm")
    async def navigation_arm(request: ArmRequest) -> dict[str, object]:
        try:
            return await runtime.navigation_arm(request.confirmation)
        except RuntimeCommandError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/navigation/cmd")
    async def navigation_command(
        request: NavigationCommandRequest,
    ) -> dict[str, object]:
        try:
            return await runtime.navigation_command(
                request.forward_mps, request.yaw_rps
            )
        except RuntimeCommandError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/navigation/stop")
    async def navigation_stop() -> dict[str, object]:
        await runtime.stop("navigation_stop")
        return await runtime.navigation_status()

    return app
