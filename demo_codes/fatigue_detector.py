import cv2


class FatigueDetector:
    def __init__(self):
        self.frame_index = 0

    def predict(self, frame):
        self.frame_index += 1

        # Replace this block with the real fatigue-detection model inference.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = gray.mean()
        confidence = min(abs(brightness - 90) / 90, 1.0)
        label = "fatigue" if brightness < 70 else "normal"

        return {
            "label": label,
            "confidence": confidence,
        }
