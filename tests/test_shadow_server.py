from collie_demo.shadow_server import RemoteDetection, parse_best_detection


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
