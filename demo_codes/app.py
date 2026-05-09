import argparse
import csv
from datetime import datetime
from pathlib import Path

import cv2

from fatigue_detector import FatigueDetector


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--log", default="fatigue_log.csv")
    return parser.parse_args()


def ensure_log(path):
    path = Path(path)
    if not path.exists():
        with path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["timestamp", "label", "confidence"])
    return path


def write_log(path, result):
    with path.open("a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                datetime.now().isoformat(timespec="seconds"),
                result["label"],
                f"{result['confidence']:.2f}",
            ]
        )


def draw_label(frame, result):
    label = f"{result['label']} ({result['confidence']:.2f})"
    color = (0, 0, 255) if result["label"] == "fatigue" else (0, 180, 0)
    cv2.rectangle(frame, (10, 10), (380, 60), (0, 0, 0), -1)
    cv2.putText(frame, label, (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


def main():
    args = parse_args()
    log_path = ensure_log(args.log)
    detector = FatigueDetector()

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"Camera {args.camera} could not be opened.")

    frame_count = 0
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break

            frame_count += 1
            result = detector.predict(frame)
            draw_label(frame, result)

            if frame_count % 10 == 0:
                write_log(log_path, result)

            cv2.imshow("TransTrack Fatigue Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
