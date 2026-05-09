from src.transtrack_demo import camera


def test_list_camera_names_returns_powershell_lines(monkeypatch):
    class Result:
        returncode = 0
        stdout = "Integrated Camera\nNVIDIA Broadcast\n"

    monkeypatch.setattr(camera.platform, "system", lambda: "Windows")
    monkeypatch.setattr(camera.subprocess, "run", lambda *args, **kwargs: Result())

    monkeypatch.setattr(camera, "_camera_names_from_directshow", lambda: [])

    assert camera.list_cameras() == [
        {"index": 0, "name": "Integrated Camera"},
        {"index": 1, "name": "NVIDIA Broadcast"},
    ]


def test_list_camera_names_returns_empty_outside_windows(monkeypatch):
    monkeypatch.setattr(camera.platform, "system", lambda: "Linux")

    monkeypatch.setattr(camera, "_camera_names_from_directshow", lambda: [])

    assert camera.list_cameras() == []


def test_choose_camera_defaults_to_zero(monkeypatch):
    monkeypatch.setattr(camera, "list_cameras", lambda: [{"index": 0, "name": "Camera A"}])
    monkeypatch.setattr("builtins.input", lambda prompt: "")

    assert camera.choose_camera() == 0


def test_choose_camera_uses_user_input(monkeypatch):
    monkeypatch.setattr(camera, "list_cameras", lambda: [{"index": 1, "name": "Camera B"}])
    monkeypatch.setattr("builtins.input", lambda prompt: "1")

    assert camera.choose_camera() == 1


def test_camera_name_returns_matching_name(monkeypatch):
    monkeypatch.setattr(camera, "list_cameras", lambda: [{"index": 2, "name": "MX Brio"}])

    assert camera.camera_name(2) == "MX Brio"
