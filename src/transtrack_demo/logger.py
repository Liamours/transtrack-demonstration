import csv
from datetime import datetime
from pathlib import Path


class CsvLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["timestamp", "label", "confidence", "class_id"])

    def write(self, result):
        with self.path.open("a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    datetime.now().isoformat(timespec="seconds"),
                    result["label"],
                    f"{result['confidence']:.4f}",
                    result["class_id"],
                ]
            )
