import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from src.transtrack_demo import live_inference


def test_save_clip_writes_all_frames(monkeypatch, tmp_path):
    written = []
    writer = MagicMock()
    writer.write.side_effect = written.append
    monkeypatch.setattr(live_inference.cv2, "VideoWriter", lambda *args: writer)
    monkeypatch.setattr(live_inference.cv2, "VideoWriter_fourcc", lambda *args: 1234)

    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)]
    live_inference.save_clip(tmp_path / "clip.mp4", frames, 10, (8, 8))

    assert len(written) == 3
    writer.release.assert_called_once()


def test_main_runs_inference_and_logs_without_real_camera(monkeypatch, tmp_path):
    model_path = tmp_path / "model.pth"
    model_path.write_bytes(b"fake")
    log_path = tmp_path / "out.csv"
    runtime_dir = tmp_path / "runtime"
    frames = [np.zeros((12, 16, 3), dtype=np.uint8) for _ in range(5)]

    capture = MagicMock()
    capture.isOpened.return_value = True
    capture.get.side_effect = lambda prop: {
        live_inference.cv2.CAP_PROP_FPS: 1,
        live_inference.cv2.CAP_PROP_FRAME_WIDTH: 16,
        live_inference.cv2.CAP_PROP_FRAME_HEIGHT: 12,
    }.get(prop, 0)
    capture.read.side_effect = [(True, frame) for frame in frames] + [(False, None)]

    monkeypatch.setattr(live_inference.cv2, "VideoCapture", lambda index: capture)
    monkeypatch.setattr(live_inference.cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(live_inference.cv2, "waitKey", lambda delay: -1)
    monkeypatch.setattr(live_inference.cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(live_inference, "save_clip", lambda *args: None)

    fake_pipeline = SimpleNamespace(
        predict=lambda clip_path, model_path: {
            "label": "normal",
            "confidence": 0.9,
            "class_id": 1,
        }
    )
    monkeypatch.setitem(sys.modules, "src.transtrack_demo.pipeline", fake_pipeline)
    monkeypatch.setattr(
        live_inference,
        "parse_args",
        lambda: SimpleNamespace(
            camera=0,
            model=str(model_path),
            clip_seconds=2,
            infer_every=2,
            log=str(log_path),
            runtime_dir=str(runtime_dir),
        ),
    )

    live_inference.main()

    assert capture.release.called
    rows = log_path.read_text().splitlines()
    assert rows[0] == "timestamp,label,confidence,class_id"
    assert len(rows) >= 2
