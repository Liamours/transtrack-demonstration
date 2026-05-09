import argparse
import os
from collections import deque
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import cv2

from . import ui
from .camera import camera_name, choose_camera
from .inference_worker import InferenceWorker

BACKENDS = {
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "any": cv2.CAP_ANY,
}


def parse_args():
    parser = argparse.ArgumentParser(description="TransTrack live fatigue demo")
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--backend", choices=BACKENDS, default="dshow")
    parser.add_argument("--model", default="models/classifier/best_val_f1.pth")
    parser.add_argument("--clip-seconds", type=int, default=20)
    parser.add_argument("--infer-every", type=int, default=10)
    parser.add_argument("--log", default="logs/fatigue_inference.csv")
    parser.add_argument("--runtime-dir", default="runtime")
    return parser.parse_args()


def draw_status(frame, result, pending_frames, needed_frames):
    if result is None:
        label = f"Collecting frames {pending_frames}/{needed_frames}"
        color = (255, 255, 255)
    else:
        warning = result["label"] in {"eyes_closed", "yawning"}
        prefix = "WARNING " if warning else ""
        label = f"{prefix}{result['label']} ({result['confidence']:.2f})"
        color = (0, 220, 255) if warning else (0, 180, 0)

    cv2.rectangle(frame, (10, 10), (520, 62), (0, 0, 0), -1)
    cv2.putText(frame, label, (24, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


def main():
    args = parse_args()
    camera_index = args.camera if args.camera is not None else choose_camera()
    selected_camera_name = camera_name(camera_index)
    model_path = Path(args.model)
    runtime_dir = Path(args.runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    capture = cv2.VideoCapture(camera_index, BACKENDS[args.backend])
    if not capture.isOpened():
        raise RuntimeError(f"Camera {camera_index} could not be opened.")

    fps = int(capture.get(cv2.CAP_PROP_FPS) or 10)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    needed_frames = max(1, fps * args.clip_seconds)
    infer_interval = max(1, fps * args.infer_every)

    frames = deque(maxlen=needed_frames)
    frame_index = 0
    worker = InferenceWorker(
        model_path=model_path,
        runtime_dir=runtime_dir,
        log_path=args.log,
        fps=fps,
        size=(width, height),
        on_result=ui.inference_line,
    )
    worker.start()

    ui.run_info(camera_index, selected_camera_name, args.backend, args.log)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame_index += 1
            frames.append(frame.copy())

            ready = len(frames) == needed_frames
            due = frame_index % infer_interval == 0
            if ready and due:
                worker.submit(frames)

            draw_status(frame, worker.latest_result, len(frames), needed_frames)
            cv2.imshow("TransTrack Live Fatigue Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        worker.stop()
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
