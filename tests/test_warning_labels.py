import numpy as np

from src.transtrack_demo import live_inference, ui


def test_terminal_warning_for_fatigue_labels(capsys):
    ui.inference_line(
        {
            "label": "eyes_closed",
            "confidence": 0.9,
            "scores": {"eyes_closed": 0.9, "normal": 0.05, "yawning": 0.05},
        }
    )

    assert "WARNING" in capsys.readouterr().out


def test_overlay_warning_for_fatigue_labels(monkeypatch):
    calls = []
    frame = np.zeros((80, 600, 3), dtype=np.uint8)

    monkeypatch.setattr(live_inference.cv2, "rectangle", lambda *args: None)
    monkeypatch.setattr(live_inference.cv2, "putText", lambda *args: calls.append(args))

    live_inference.draw_status(
        frame,
        {"label": "yawning", "confidence": 0.8},
        pending_frames=20,
        needed_frames=20,
    )

    assert "WARNING yawning" in calls[0][1]
