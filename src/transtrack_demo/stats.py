WARNING_LABELS = {"eyes_closed", "yawning"}


class FatigueStats:
    def __init__(self):
        self.total = 0
        self.warning = 0
        self.counts = {"eyes_closed": 0, "normal": 0, "yawning": 0}

    def update(self, result):
        label = result["label"]
        self.total += 1
        self.counts[label] = self.counts.get(label, 0) + 1
        if label in WARNING_LABELS:
            self.warning += 1

    @property
    def fatigue_rate(self):
        if self.total == 0:
            return 0.0
        return self.warning / self.total

    def as_dict(self):
        return {
            "total": self.total,
            "warning": self.warning,
            "fatigue_rate": self.fatigue_rate,
            "counts": dict(self.counts),
        }
