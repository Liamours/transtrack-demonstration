import queue
import threading
from pathlib import Path

import cv2

from .logger import CsvLogger


def save_clip(path, frames, fps, size):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    for frame in frames:
        writer.write(frame)
    writer.release()


class InferenceWorker:
    def __init__(self, model_path, runtime_dir, log_path, fps, size, on_result=None):
        self.model_path = Path(model_path)
        self.runtime_dir = Path(runtime_dir)
        self.log = CsvLogger(log_path)
        self.fps = fps
        self.size = size
        self.on_result = on_result
        self.tasks = queue.Queue(maxsize=1)
        self.latest_result = None
        self.running = False
        self.thread = None

    def start(self):
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, frames):
        if self.tasks.full():
            return False
        self.tasks.put_nowait(list(frames))
        return True

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

    def _run(self):
        from .pipeline import predict

        while self.running:
            try:
                frames = self.tasks.get(timeout=0.1)
            except queue.Empty:
                continue

            clip_path = self.runtime_dir / "latest_camera_clip.mp4"
            save_clip(clip_path, frames, self.fps, self.size)
            result = predict(clip_path, self.model_path)
            self.latest_result = result
            self.log.write(result)
            if self.on_result:
                self.on_result(result)
            self.tasks.task_done()
