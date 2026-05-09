import cv2


def list_cameras(limit=5):
    cameras = []
    for index in range(limit):
        capture = cv2.VideoCapture(index)
        if capture.isOpened():
            cameras.append(index)
        capture.release()
    return cameras


def choose_camera(limit=5):
    cameras = list_cameras(limit)
    if cameras:
        print(f"Available cameras: {', '.join(str(c) for c in cameras)}")
    else:
        print("No camera detected by quick scan. You can still try index 0.")

    raw = input("Choose camera index [0]: ").strip()
    return int(raw) if raw else 0
