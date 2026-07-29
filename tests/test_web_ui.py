from pathlib import Path


def test_one_click_follow_is_owned_by_robot_service() -> None:
    html = (Path(__file__).parents[1] / "web" / "index.html").read_text()

    assert "setInterval(pulse" not in html
    assert "pulseLoopRunning" not in html
    assert "while(following)" not in html
    assert "await api('/api/pulse')" not in html
    assert "follow.onclick=beginFollow" in html
    assert "await api('/api/follow',{confirmation:'TARGET AND PATH CLEAR'})" in html
    assert "Robot-side follow is active" in html
    assert "PRESS AND HOLD" not in html
    assert "pointerdown" not in html
    assert "addEventListener('blur',endFollow)" not in html
    assert "visibilitychange" not in html
    assert "addEventListener('pagehide',endFollow)" in html
    assert "keepalive:true" in html


def test_each_detection_gets_a_select_control() -> None:
    html = (Path(__file__).parents[1] / "web" / "index.html").read_text()

    assert "for(const [index,item] of state.detections.entries())" in html
    assert "choose.onclick=()=>chooseTarget(item.label,item.center)" in html
    assert "await api('/api/target',{target:name,center})" in html
    assert "closestSelectedIndex" in html
    assert "FOLLOW SELECTED FRUIT" in html
    assert "Blue whale" not in html
    assert "Yellow whale" not in html
    assert "SpeechRecognition" not in html


def test_camera_uses_one_low_latency_stream_with_browser_overlay() -> None:
    html = (Path(__file__).parents[1] / "web" / "index.html").read_text()

    assert "if(refreshInFlight)return" in html
    assert "finally{refreshInFlight=false}" in html
    assert 'src="/camera-stream.mjpg"' in html
    assert 'id="camera-overlay"' in html
    assert "renderOverlay(s)" in html
    assert "s.camera_fps" in html
    assert "camera.jpg?t=${Date.now()}" not in html


def test_tracker_confidence_is_not_presented_as_live_yolo_confidence() -> None:
    html = (Path(__file__).parents[1] / "web" / "index.html").read_text()

    assert "tracker awaiting YOLO check" in html
    assert "observation.confidence===null" in html
    assert "revalidation_failures" in html
    assert "revalidation_failures_required" in html


def test_stage_health_is_visible_in_the_ui() -> None:
    html = (Path(__file__).parents[1] / "web" / "index.html").read_text()

    assert "STAGE READY" in html
    assert "failedHealth" in html
    assert "gpu_ready" not in html  # rendered generically from the health object
    assert "YOLO verified" in html
    assert "misses ${misses}/${required}" in html
    assert "WAITING FOR STABLE TRACK" in html
    assert "WAITING FOR YOLO" in html
    assert "WAITING FOR FRESH FRAME" in html
    assert "s.follow_readiness" in html


def test_memory_demo_ui_keeps_stop_and_manual_fallback() -> None:
    html = (Path(__file__).parents[1] / "web" / "index.html").read_text()

    assert "SAVE CLASS" in html
    assert "RUN REMEMBER & FIND" in html
    assert "direct yaw" in html
    assert "await api('/api/memory/capture',{target:name,center,round_id:roundId})" in html
    assert "await api('/api/demo/start',{confirmation:'TARGET SAVED AND AREA CLEAR'})" in html
    assert "await api('/api/demo/go',{confirmation:'CLASS LOCKED AND PATH CLEAR'})" in html
    assert 'id="go" disabled' in html
    assert "go.onclick=approveGo" in html
    assert "WAITING FOR CLASS" in html
    assert "GO TO FRUIT" in html
    assert "await api('/api/round/reset')" in html
    assert "START OVER (FULL RESET)" in html
    assert "uiRoundGeneration+=1" in html
    assert "Manual follower fallback" in html
    assert "FIND THIS ${saved.label.toUpperCase()} & RETURN" in html
    assert "return_home_status" in html
    assert "search_progress_deg" in html
    assert "search_sweep_deg" in html
    assert "detector_confidence" in html
    assert "No saved-class detection yet" in html
    assert "exact physical prop" not in html
    assert "exact instance" not in html
    assert "Candidate similarity" not in html
    assert "FIND A DIFFERENT" not in html
    assert "Fruit to reject" not in html
    assert 'id="stop"' in html
    assert "following||demoActive||pointingActive||startingFollow" in html


def test_touch_range_is_calibrated_as_measured_distance() -> None:
    html = (Path(__file__).parents[1] / "web" / "index.html").read_text()

    assert "Touch-range calibration" in html
    assert 'id="final-distance"' in html
    assert 'min="2" max="30"' in html
    assert "Extra measured travel" in html
    assert "final_approach_measured_distance_m" in html
    assert "arrival_pointing_status" in html
    assert "contact_status" in html
    assert (
        "await api('/api/calibration/final-approach',{distance_m:cm/100})"
        in html
    )


def test_voice_mission_ui_exposes_live_mic_and_emergency_controls() -> None:
    html = (Path(__file__).parents[1] / "web" / "index.html").read_text()

    assert "Say just: “apple”, “banana”, or “pear”" in html
    assert "without the Hello or Stretch gestures" in html
    assert "lie down and bark for five seconds" in html
    assert "listen for the next fruit" in html
    assert 'id="voice-state"' in html
    assert 'id="voice-live"' in html
    assert 'id="voice-start"' in html
    assert 'id="voice-stop"' in html
    assert 'id="voice-bark"' in html
    assert ":8098/api/status" not in html  # assembled from the shared origin
    assert "webrtc_connected" in html
    assert "scribe_connected" in html
    assert "last_partial" in html
    assert "mic_source" in html
    assert "mission_busy" in html
    assert "arrival_bark_status" in html
    assert "voiceStart.onclick" in html
    assert "voiceStop.onclick" in html
    assert "voiceBark.onclick" in html


def test_live_pointing_ui_requires_standdown_and_keeps_guarded_stop() -> None:
    html = (Path(__file__).parents[1] / "web" / "index.html").read_text()

    assert "Show the real pointing policy" in html
    assert 'id="pointing-prepare"' in html
    assert 'id="pointing-run"' in html
    assert 'id="pointing-stop"' in html
    assert "WOOF IS CLEAR TO LIE DOWN" in html
    assert "WOOF IS LYING DOWN AND TARGET AREA IS CLEAR" in html
    assert "await api(path,confirmation?{confirmation}:undefined)" in html
    assert "renderPointing(s)" in html
    assert "There is no bypass button." in html
    assert "--bypass-roll-guard" not in html


def test_mission_ui_describes_arrival_rest_and_return_home() -> None:
    html = (Path(__file__).parents[1] / "web" / "index.html").read_text()

    assert "lies down for five seconds, stands" in html
    assert "arrival_rest_status" in html
    assert "lay-down" in html
