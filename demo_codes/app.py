from fatigue_detector import FatigueDetector


def main():
    detector = FatigueDetector(yawn_threshold=3, eye_closed_threshold=4)
    sample_frames = [
        {"eye_closed": False, "yawning": False},
        {"eye_closed": True, "yawning": False},
        {"eye_closed": True, "yawning": True},
        {"eye_closed": True, "yawning": True},
        {"eye_closed": True, "yawning": True},
    ]

    for index, frame in enumerate(sample_frames, start=1):
        result = detector.analyze(frame)
        print(f"Frame {index}: {result['status']} | score={result['score']}")


if __name__ == "__main__":
    main()
