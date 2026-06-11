import os

import cv2
import mediapipe as mp
import numpy as np


class FaceLandmarkDetector:
    """Compatibility wrapper for MediaPipe legacy FaceMesh and Tasks FaceLandmarker."""

    def __init__(self, max_num_faces=1):
        self.mode = None
        self.legacy_face_mesh = None
        self.tasks_landmarker = None

        if hasattr(mp, "solutions"):
            self._init_legacy(max_num_faces)
        else:
            self._init_tasks(max_num_faces)

    def _init_legacy(self, max_num_faces):
        self.mode = "legacy"
        self.legacy_face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def _init_tasks(self, max_num_faces):
        self.mode = "tasks"
        model_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "models", "face_landmarker.task")
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MediaPipe face landmarker model not found: {model_path}")

        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=max_num_faces,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.tasks_landmarker = FaceLandmarker.create_from_options(options)

    def process(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.mode == "legacy":
            results = self.legacy_face_mesh.process(rgb_frame)
            return results.multi_face_landmarks or []

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb_frame),
        )
        result = self.tasks_landmarker.detect(mp_image)
        return result.face_landmarks or []


def get_landmark(face_landmarks, index):
    if hasattr(face_landmarks, "landmark"):
        return face_landmarks.landmark[index]
    return face_landmarks[index]
