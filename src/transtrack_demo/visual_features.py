import cv2
import numpy as np

EYE_MOUTH_INDICES = [33, 159, 158, 133, 153, 144, 263, 386, 385, 362, 380, 373, 13, 14, 61, 291]


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
            "landmarks": [landmarks[index] for index in EYE_MOUTH_INDICES],
            "ear": ear,
            "mar": None if np.isnan(mar) else mar,
        }


def draw_landmarks(frame, landmarks):
    height, width = frame.shape[:2]
    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
    return frame


def draw_ear_mar(frame, features):
    ear = features.get("ear") if features else None
    mar = features.get("mar") if features else None
    ear_text = "EAR: -" if ear is None else f"EAR: {ear:.4f}"
    mar_text = "MAR: -" if mar is None else f"MAR: {mar:.4f}"

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (190, 72), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, ear_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, mar_text, (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return frame


def draw_stats(frame, stats):
    data = stats.as_dict()
    lines = [
        f"Inferences: {data['total']}",
        f"Warnings: {data['warning']}",
        f"Fatigue Rate: {data['fatigue_rate'] * 100:.1f}%",
        f"Normal: {data['counts'].get('normal', 0)}",
    ]
    height, width = frame.shape[:2]
    box_width = 250
    box_height = 132
    x1 = max(10, width - box_width - 10)
    y1 = 10
    x2 = width - 10
    y2 = y1 + box_height

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for index, line in enumerate(lines):
        y = y1 + 30 + index * 27
        cv2.putText(frame, line, (x1 + 14, y), cv2.FONT_HERSHEY_DUPLEX, 0.62, (255, 255, 255), 1)
    return frame


def zoom_landmark_region(frame, landmarks, scale=2.5, padding=80):
    height, width = frame.shape[:2]
    if not landmarks:
        crop_width = max(1, width // 2)
        crop_height = max(1, height // 2)
        x1 = (width - crop_width) // 2
        y1 = (height - crop_height) // 2
        crop = frame[y1:y1 + crop_height, x1:x1 + crop_width]
    else:
        points = np.array(
            [[int(landmark.x * width), int(landmark.y * height)] for landmark in landmarks],
            dtype=np.int32,
        )
        x1 = max(0, int(points[:, 0].min()) - padding)
        y1 = max(0, int(points[:, 1].min()) - padding)
        x2 = min(width, int(points[:, 0].max()) + padding)
        y2 = min(height, int(points[:, 1].max()) + padding)
        crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return frame.copy()

    zoom_width = max(1, int(crop.shape[1] * scale))
    zoom_height = max(1, int(crop.shape[0] * scale))
    return cv2.resize(crop, (zoom_width, zoom_height), interpolation=cv2.INTER_LINEAR)
