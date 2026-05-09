from unittest.mock import MagicMock

from src.transtrack_demo import camera


def test_list_cameras_returns_open_indices(monkeypatch):
    captures = []

    def fake_capture(index):
        capture = MagicMock()
        capture.isOpened.return_value = index in {0, 2}
        captures.append(capture)
        return capture

    monkeypatch.setattr(camera.cv2, "VideoCapture", fake_capture)

    assert camera.list_cameras(limit=3) == [0, 2]
    assert all(capture.release.called for capture in captures)


def test_choose_camera_defaults_to_zero(monkeypatch):
    monkeypatch.setattr(camera, "list_cameras", lambda limit=5: [0, 1])
    monkeypatch.setattr("builtins.input", lambda prompt: "")

    assert camera.choose_camera() == 0


def test_choose_camera_uses_user_input(monkeypatch):
    monkeypatch.setattr(camera, "list_cameras", lambda limit=5: [0, 1])
    monkeypatch.setattr("builtins.input", lambda prompt: "1")

    assert camera.choose_camera() == 1
