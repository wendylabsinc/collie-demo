from pathlib import Path


def test_operator_ui_is_minimal_and_records_each_activation() -> None:
    html = (Path(__file__).parents[1] / "web" / "index.html").read_text()

    assert "ACTIVATE DEMO" in html
    assert "STOP NOW" in html
    assert 'href="/debug"' in html
    assert 'href="/tests"' in html
    assert "collie-demo-run-history-v1" in html
    assert "startRun()" in html
    assert "compactSnapshot" in html
    assert "updateRun(demo,voice)" in html
    assert "/api/listen/start" in html


def test_debug_ui_displays_and_exports_stage_results() -> None:
    html = (Path(__file__).parents[1] / "web" / "debug.html").read_text()

    assert "Recorded demo runs" in html
    assert 'id="run-history"' in html
    assert 'id="export-runs"' in html
    assert 'id="clear-runs"' in html
    assert "renderRunHistory" in html
    assert "record.stages" in html
    assert "EXPORT RUNS JSON" in html


def test_reusable_mini_test_index_lists_existing_tools() -> None:
    root = Path(__file__).parents[1] / "web" / "tests"
    html = (root / "index.html").read_text()

    assert (root / "forward-motion.html").is_file()
    assert (root / "nav2-forward-motion.html").is_file()
    assert (root / "fruit-detector.html").is_file()
    assert 'href="/tests/forward-motion"' in html
    assert 'href="/tests/nav2-forward-motion"' in html
    assert 'href="/tests/fruit-detector"' in html


def test_one_click_follow_is_owned_by_robot_service() -> None:
    html = (Path(__file__).parents[1] / "web" / "debug.html").read_text()

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
    html = (Path(__file__).parents[1] / "web" / "debug.html").read_text()

    assert "for(const [index,item] of state.detections.entries())" in html
    assert "choose.onclick=()=>chooseTarget(item.label,item.center)" in html
    assert "await api('/api/target',{target:name,center})" in html
    assert "closestSelectedIndex" in html
    assert "FOLLOW SELECTED FRUIT" in html
    assert "Blue whale" not in html
    assert "Yellow whale" not in html
    assert "SpeechRecognition" not in html


def test_camera_uses_one_low_latency_stream_with_browser_overlay() -> None:
    html = (Path(__file__).parents[1] / "web" / "debug.html").read_text()

    assert "if(refreshInFlight)return" in html
    assert "finally{refreshInFlight=false}" in html
    assert 'src="/camera-stream.mjpg"' in html
    assert 'id="camera-overlay"' in html
    assert "renderOverlay(s)" in html
    assert "s.camera_fps" in html
    assert "camera.jpg?t=${Date.now()}" not in html


def test_tracker_confidence_is_not_presented_as_live_yolo_confidence() -> None:
    html = (Path(__file__).parents[1] / "web" / "debug.html").read_text()

    assert "tracker awaiting YOLO check" in html
    assert "observation.confidence===null" in html
    assert "revalidation_failures" in html
    assert "revalidation_failures_required" in html


def test_stage_health_is_visible_in_the_ui() -> None:
    html = (Path(__file__).parents[1] / "web" / "debug.html").read_text()

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
    html = (Path(__file__).parents[1] / "web" / "debug.html").read_text()

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
    assert 'id="nav-status"' in html
    assert "Return planner: Nav2 map frame" in html
    assert "local odometry fallback" in html
    assert "no global obstacle-detour planning" in html
    assert "nav2_health" in html
    assert "scan_healthy" in html
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


def test_offscreen_final_push_is_fixed_and_reported() -> None:
    html = (Path(__file__).parents[1] / "web" / "debug.html").read_text()

    assert "Off-screen final push" in html
    assert "1.0 m/s forward push for 0.4 seconds" in html
    assert 'id="final-distance"' not in html
    assert "APPLY TOUCH RANGE" not in html
    assert "final_approach_measured_distance_m" in html
    assert "final_push_duration_s" in html
    assert "arrival_pointing_status" in html
    assert "contact_status" in html
    assert "/api/calibration/final-approach" not in html


def test_voice_mission_ui_exposes_live_mic_and_emergency_controls() -> None:
    html = (Path(__file__).parents[1] / "web" / "debug.html").read_text()

    assert "Say just: “apple”, “banana”, or “pear”" in html
    assert "without the Hello or Stretch gestures" in html
    assert "lie down for five seconds, stand up" in html
    assert (
        "follow the configured return planner to the captured start pose and heading"
        in html
    )
    assert "listen for the next fruit" in html
    assert 'id="voice-state"' in html
    assert 'id="voice-live"' in html
    assert 'id="voice-start"' in html
    assert 'id="voice-stop"' in html
    assert 'id="voice-bark"' in html
    assert 'id="voice-command-input"' in html
    assert 'id="voice-submit"' in html
    assert "Type a fruit instead of speaking" in html
    assert "RUN FULL SEQUENCE" in html
    assert "approach → lie down/stand up → planned return Home" in html
    assert "No additional Go click is required." in html
    assert "No more input is needed; Woof will return Home on its own." in html
    assert "voice_mission_complete_ready" in html
    assert "voiceApi('/api/command',{command:fruit})" in html
    assert "voiceCommandForm.onsubmit=submitTypedFruit" in html
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


def test_live_pointing_ui_prepares_standing_policy_and_keeps_guarded_stop() -> None:
    html = (Path(__file__).parents[1] / "web" / "debug.html").read_text()

    assert "Experimental pointing policy (manual only)" in html
    assert 'id="pointing-prepare"' in html
    assert 'id="pointing-run"' in html
    assert 'id="pointing-stop"' in html
    assert "WOOF IS CLEAR FOR STANDING POINT" in html
    assert "AREA IS CLEAR AND WOOF MAY MOVE" in html
    assert "RUN 6.0s STANDING POINT" in html
    assert "await api(path,confirmation?{confirmation}:undefined)" in html
    assert "renderPointing(s)" in html
    assert "There is no bypass button." in html
    assert "--bypass-roll-guard" not in html


def test_mission_ui_describes_stock_posture_and_configured_return_home() -> None:
    html = (Path(__file__).parents[1] / "web" / "debug.html").read_text()

    assert "lie down for five seconds, stand up" in html
    assert "saved start pose and heading" in html
    assert "home_pose_validation" in html
    assert "maximum_position_span_m" in html
    assert "arrival_pointing_status" in html


def test_stage_image_uses_stock_posture_instead_of_autonomous_pointing() -> None:
    dockerfile = (Path(__file__).parents[1] / "Dockerfile").read_text()

    assert "COLLIE_ARRIVAL_REST_ENABLED=1" in dockerfile
    assert "COLLIE_ARRIVAL_REST_DURATION_S=5.0" in dockerfile
    assert "COLLIE_ARRIVAL_POINTING_ENABLED=0" in dockerfile
    assert "COLLIE_RETURN_POSE_CAPTURE_DURATION_S=0.50" in dockerfile
    assert "COLLIE_RETURN_HOME_ENABLED=1" in dockerfile
    assert "COLLIE_RETURN_ARRIVAL_TOLERANCE_M=0.10" in dockerfile
    assert "COLLIE_RETURN_HEADING_TOLERANCE_DEG=5.0" in dockerfile


def test_forward_calibration_ui_is_tiny_explicit_and_auto_stopping() -> None:
    html = (
        Path(__file__).parents[1] / "web" / "tests" / "forward-motion.html"
    ).read_text()

    assert "Direction" in html
    assert "forward" in html
    assert "Amount" in html
    assert "Movement?" in html
    assert "MOVEMENT? YES" in html
    assert "MOVEMENT? NO" in html
    assert 'type="number" step="any"' in html
    assert 'min="' not in html
    assert 'max="0.15"' not in html
    assert "Math.min(.15" not in html
    assert "x.amount.toFixed" not in html
    assert "pendingAmount+.01" not in html
    assert "One 0.4 second pulse" in html
    assert "PATH CLEAR AND STOP READY" in html
    assert "'/api/calibration/forward-pulse'" in html
    assert "result.stopped" in html
    assert "await api('/api/stop')" in html
    assert "navigator.sendBeacon('/api/stop')" in html


def test_nav2_forward_calibration_ui_is_explicit_about_direct_motion() -> None:
    html = (
        Path(__file__).parents[1]
        / "web"
        / "tests"
        / "nav2-forward-motion.html"
    ).read_text()

    assert "Nav2 direct forward deadband" in html
    assert "direct SportClient" in html
    assert "NO FACTORY OBSTACLE AVOIDANCE" in html
    assert "NO APP SPEED CLAMP" in html
    assert "STEP TAKEN? YES" in html
    assert "NO — POSTURE ONLY" in html
    assert "Repeat the exact same amount" in html
    assert 'id="ready"' not in html
    assert "pulse.disabled=running||!validAmount()" in html
    assert "yes.disabled" not in html
    assert "no.disabled" not in html
    assert "PATH CLEAR NO AVOIDANCE STOP READY" in html
    assert "'/api/calibration/nav2-forward-pulse'" in html
    assert "result.amount_mps" in html
    assert "result.stopped" in html
    assert "navigator.sendBeacon('/api/stop')" in html
