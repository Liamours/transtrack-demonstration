class FatigueDetector:
    def __init__(self, yawn_threshold=3, eye_closed_threshold=4):
        self.yawn_threshold = yawn_threshold
        self.eye_closed_threshold = eye_closed_threshold
        self.yawn_count = 0
        self.eye_closed_count = 0

    def analyze(self, frame):
        self.yawn_count = self._update_count(self.yawn_count, frame["yawning"])
        self.eye_closed_count = self._update_count(
            self.eye_closed_count,
            frame["eye_closed"],
        )

        score = self.yawn_count + self.eye_closed_count
        status = "fatigue_detected" if self._is_fatigued() else "normal"

        return {
            "status": status,
            "score": score,
            "yawn_count": self.yawn_count,
            "eye_closed_count": self.eye_closed_count,
        }

    def _is_fatigued(self):
        return (
            self.yawn_count >= self.yawn_threshold
            or self.eye_closed_count >= self.eye_closed_threshold
        )

    @staticmethod
    def _update_count(current, detected):
        return current + 1 if detected else 0
