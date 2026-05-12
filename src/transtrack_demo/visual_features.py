import cv2
import numpy as np


class VisualFeatureExtractor:
    def __init__(self):
        from .pipeline import _ensure_mediapipe_model
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions

        model_path = _ensure_mediapipe_model()
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        self.mp = mp
        self.landmarker = FaceLandmarker.create_from_options(options)

    def close(self):
        self.landmarker.close()

    def analyze(self, frame):
        from .pipeline import _LEFT_EYE, _MOUTH, _RIGHT_EYE, _ear, _mar
        from mediapipe.tasks.python.vision.core.image import Image as MpImage

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.landmarker.detect(MpImage(image_format=self.mp.ImageFormat.SRGB, data=rgb))
        if not result.face_landmarks:
            return {"landmarks": [], "ear": None, "mar": None}

        landmarks = result.face_landmarks[0]
        ear_l = _ear(landmarks, _LEFT_EYE)
        ear_r = _ear(landmarks, _RIGHT_EYE)
        ear_values = [value for value in (ear_l, ear_r) if not np.isnan(value)]
        ear = float(np.mean(ear_values)) if ear_values else None
        mar = _mar(landmarks, _MOUTH)

        return {
            "landmarks": landmarks,
            "ear": ear,
            "mar": None if np.isnan(mar) else mar,
        }


def draw_landmarks(frame, landmarks):
    height, width = frame.shape[:2]
    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)
    return frame
