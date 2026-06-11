from datetime import datetime

import cv2
import numpy as np
import torch
from ultralytics import YOLO


class ObjectDetector:
    PHONE_LABELS = {"cell phone", "mobile phone", "phone", "smartphone", "cellphone"}
    BOOK_LABELS = {"book", "textbook", "notebook"}

    def __init__(self, config):
        self.config = config["detection"]["objects"]
        self.model = None
        self.alert_logger = None
        self.min_confidence = float(self.config.get("min_confidence", 0.45))
        self.phone_min_confidence = float(self.config.get("phone_min_confidence", 0.35))
        self.book_min_confidence = float(self.config.get("book_min_confidence", self.min_confidence))
        self.max_fps = max(float(self.config.get("max_fps", 5)), 0.1)
        self.last_detection_time = datetime.min
        self._initialize_model()

    def _initialize_model(self):
        """Initialize YOLO and run one warm-up inference."""
        try:
            self.model = YOLO("models/yolov8n.pt")
            imgsz = int(self.config.get("imgsz", 640))

            self.model.overrides["conf"] = self.min_confidence
            self.model.overrides["device"] = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.overrides["imgsz"] = imgsz
            self.model.overrides["iou"] = 0.45

            self.model(np.zeros((imgsz, imgsz, 3), dtype=np.uint8), verbose=False)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize object detector: {str(e)}")

    def set_alert_logger(self, alert_logger):
        self.alert_logger = alert_logger

    def _log_alert(self, alert_type, message):
        if self.alert_logger:
            self.alert_logger.log_alert(alert_type, message)

    def _should_skip_frame(self, current_time):
        elapsed = (current_time - self.last_detection_time).total_seconds()
        return elapsed < (1.0 / self.max_fps)

    def _resize_for_inference(self, frame):
        if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
            raise ValueError("Invalid frame supplied to object detector")

        orig_h, orig_w = frame.shape[:2]
        if orig_h <= 0 or orig_w <= 0:
            raise ValueError("Invalid frame size supplied to object detector")

        new_w = int(self.config.get("inference_width", 640))
        if new_w <= 0:
            new_w = 640
        new_h = max(1, int(orig_h * (new_w / orig_w)))

        return cv2.resize(frame, (new_w, new_h)), (orig_w, orig_h, new_w, new_h)

    def _class_name(self, result, cls):
        names = getattr(result, "names", None) or getattr(self.model, "names", {})

        if isinstance(names, dict):
            return str(names.get(cls, cls)).strip().lower()

        if isinstance(names, (list, tuple)) and 0 <= cls < len(names):
            return str(names[cls]).strip().lower()

        return str(cls)

    def _forbidden_label(self, class_name):
        normalized = " ".join(class_name.replace("_", " ").replace("-", " ").split())

        if normalized in self.PHONE_LABELS:
            return "cell phone"

        if normalized in self.BOOK_LABELS:
            return "book"

        return None

    def _confidence_threshold(self, label):
        if label == "cell phone":
            return self.phone_min_confidence
        if label == "book":
            return self.book_min_confidence
        return self.min_confidence

    def _draw_detection(self, frame, box, label, confidence, scale):
        orig_w, orig_h, new_w, new_h = scale
        x1, y1, x2, y2 = np.asarray(box.xyxy[0], dtype=float)

        x1 = max(0, min(orig_w - 1, int(x1 * (orig_w / new_w))))
        y1 = max(0, min(orig_h - 1, int(y1 * (orig_h / new_h))))
        x2 = max(0, min(orig_w - 1, int(x2 * (orig_w / new_w))))
        y2 = max(0, min(orig_h - 1, int(y2 * (orig_h / new_h))))

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            frame,
            f"{label} {confidence:.2f}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )

    def detect_objects(self, frame, visualize=False):
        """Return True when a prohibited object is detected in the frame."""
        current_time = datetime.now()
        if self._should_skip_frame(current_time):
            return False

        try:
            resized_frame, scale = self._resize_for_inference(frame)
            results = self.model(resized_frame, verbose=False)

            detected = False
            for result in results:
                for box in getattr(result, "boxes", []):
                    cls = int(box.cls)
                    confidence = float(box.conf)
                    label = self._forbidden_label(self._class_name(result, cls))

                    if not label or confidence < self._confidence_threshold(label):
                        continue

                    detected = True
                    self._log_alert(
                        "FORBIDDEN_OBJECT",
                        f"Detected {label} with confidence {confidence:.2f}",
                    )

                    if visualize:
                        self._draw_detection(frame, box, label, confidence, scale)

            self.last_detection_time = current_time
            return detected
        except Exception as e:
            self.last_detection_time = current_time
            self._log_alert("OBJECT_DETECTION_ERROR", f"Object detection failed: {str(e)}")
            return False
