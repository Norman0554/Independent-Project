import os
import sys
import unittest
from datetime import datetime, timedelta

import numpy as np


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
sys.path.insert(0, SRC_DIR)

from detection.object_detection import ObjectDetector


def object_config():
    return {
        "detection": {
            "objects": {
                "min_confidence": 0.45,
                "phone_min_confidence": 0.35,
                "book_min_confidence": 0.45,
                "max_fps": 5,
                "inference_width": 640,
                "imgsz": 640,
            }
        }
    }


class FakeLogger:
    def __init__(self):
        self.entries = []

    def log_alert(self, alert_type, message):
        self.entries.append((alert_type, message))


class FakeBox:
    def __init__(self, cls, confidence, xyxy=None):
        self.cls = cls
        self.conf = confidence
        self.xyxy = np.array([xyxy or [10, 20, 120, 180]], dtype=float)


class FakeResult:
    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names


class FakeModel:
    def __init__(self, boxes=None, names=None, error=None):
        self.boxes = boxes or []
        self.names = names or {}
        self.error = error
        self.call_count = 0

    def __call__(self, frame, verbose=False):
        self.call_count += 1
        if self.error:
            raise RuntimeError(self.error)
        return [FakeResult(self.boxes, self.names)]


class TestableObjectDetector(ObjectDetector):
    def __init__(self, config, model):
        self.fake_model = model
        super().__init__(config)

    def _initialize_model(self):
        self.model = self.fake_model


class ObjectDetectorTests(unittest.TestCase):
    def make_detector(self, model):
        detector = TestableObjectDetector(object_config(), model)
        detector.last_detection_time = datetime.now() - timedelta(seconds=1)
        detector.set_alert_logger(FakeLogger())
        return detector

    def test_detects_phone_with_phone_specific_threshold(self):
        model = FakeModel(
            boxes=[FakeBox(67, 0.40)],
            names={67: "cell phone"},
        )
        detector = self.make_detector(model)

        detected = detector.detect_objects(np.zeros((480, 640, 3), dtype=np.uint8))

        self.assertTrue(detected)
        self.assertEqual(detector.alert_logger.entries[0][0], "FORBIDDEN_OBJECT")
        self.assertIn("cell phone", detector.alert_logger.entries[0][1])

    def test_rejects_phone_below_threshold(self):
        model = FakeModel(
            boxes=[FakeBox(67, 0.34)],
            names={67: "cell phone"},
        )
        detector = self.make_detector(model)

        detected = detector.detect_objects(np.zeros((480, 640, 3), dtype=np.uint8))

        self.assertFalse(detected)
        self.assertEqual(detector.alert_logger.entries, [])

    def test_detects_book_from_list_names_and_draws_warning(self):
        names = ["unknown"] * 74
        names[73] = "book"
        model = FakeModel(
            boxes=[FakeBox(73, 0.50)],
            names=names,
        )
        detector = self.make_detector(model)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        detected = detector.detect_objects(frame, visualize=True)

        self.assertTrue(detected)
        self.assertGreater(int(frame.sum()), 0)

    def test_skips_detection_when_max_fps_limit_has_not_elapsed(self):
        model = FakeModel(
            boxes=[FakeBox(67, 0.90)],
            names={67: "cell phone"},
        )
        detector = TestableObjectDetector(object_config(), model)
        detector.last_detection_time = datetime.now()

        detected = detector.detect_objects(np.zeros((480, 640, 3), dtype=np.uint8))

        self.assertFalse(detected)
        self.assertEqual(model.call_count, 0)

    def test_logs_and_recovers_from_model_error(self):
        model = FakeModel(error="model failed")
        detector = self.make_detector(model)

        detected = detector.detect_objects(np.zeros((480, 640, 3), dtype=np.uint8))

        self.assertFalse(detected)
        self.assertEqual(detector.alert_logger.entries[0][0], "OBJECT_DETECTION_ERROR")
        self.assertIn("model failed", detector.alert_logger.entries[0][1])


if __name__ == "__main__":
    unittest.main()
