from src.transtrack_demo import camera


def test_list_camera_names_returns_powershell_lines(monkeypatch):
    class Result:
        returncode = 0
        stdout = "Integrated Camera\nNVIDIA Broadcast\n"

    monkeypatch.setattr(camera.platform, "system", lambda: "Windows")
    monkeypatch.setattr(camera.subprocess, "run", lambda *args, **kwargs: Result())

    assert camera.list_camera_names() == ["Integrated Camera", "NVIDIA Broadcast"]


def test_list_camera_names_returns_empty_outside_windows(monkeypatch):
    monkeypatch.setattr(camera.platform, "system", lambda: "Linux")

    assert camera.list_camera_names() == []


def test_choose_camera_defaults_to_zero(monkeypatch):
    monkeypatch.setattr(camera, "list_camera_names", lambda: ["Camera A", "Camera B"])
    monkeypatch.setattr("builtins.input", lambda prompt: "")

    assert camera.choose_camera() == 0


def test_choose_camera_uses_user_input(monkeypatch):
    monkeypatch.setattr(camera, "list_camera_names", lambda: ["Camera A", "Camera B"])
    monkeypatch.setattr("builtins.input", lambda prompt: "1")

    assert camera.choose_camera() == 1
