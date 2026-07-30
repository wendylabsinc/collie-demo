from collie_demo.shadow_server import (
    RemoteDetection,
    compare_boxes,
    parse_best_detection,
    parse_detections,
)


def test_parse_best_detection_selects_highest_confidence() -> None:
    payload = {
        "produce": {
            "detections": [
                {
                    "label": "pear",
                    "confidence": 0.72,
                    "bbox_xyxy": [10, 20, 40, 70],
                },
                {
                    "label": "apple",
                    "confidence": 0.91,
                    "bbox_xyxy": [100, 120, 180, 240],
                },
            ]
        }
    }

    assert parse_best_detection(payload) == RemoteDetection(
        label="apple",
        confidence=0.91,
        bbox_xyxy=(100, 120, 180, 240),
    )


def test_parse_best_detection_ignores_invalid_boxes() -> None:
    payload = {
        "produce": {
            "detections": [
                {
                    "label": "pear",
                    "confidence": 0.99,
                    "bbox_xyxy": [40, 20, 10, 70],
                }
            ]
        }
    }

    assert parse_best_detection(payload) is None


def test_parse_detections_returns_confidence_order() -> None:
    payload = {
        "produce": {
            "detections": [
                {
                    "label": "banana",
                    "confidence": 0.31,
                    "bbox_xyxy": [10, 20, 40, 70],
                },
                {
                    "label": "pear",
                    "confidence": 0.82,
                    "bbox_xyxy": [100, 120, 180, 240],
                },
            ]
        }
    }

    assert [item.label for item in parse_detections(payload)] == ["pear", "banana"]


def test_compare_boxes_reports_iou_and_center_delta() -> None:
    detection = RemoteDetection(
        label="pear",
        confidence=0.9,
        bbox_xyxy=(10, 20, 50, 80),
    )

    comparison = compare_boxes(
        detection,
        (20.0, 20.0, 40.0, 60.0),
        tracker_label="pear",
    )

    assert comparison["availability"] == "both_available"
    assert comparison["same_label"] is True
    assert comparison["iou"] == 0.6
    assert comparison["center_delta_px"] == 10.0


def test_compare_boxes_distinguishes_detector_gap() -> None:
    comparison = compare_boxes(
        None,
        (20.0, 20.0, 40.0, 60.0),
        tracker_label="pear",
    )

    assert comparison == {
        "availability": "tracker_only_detector_gap",
        "same_label": None,
        "iou": None,
        "center_delta_px": None,
    }
