import platform
import subprocess

from . import ui


def _camera_names_from_directshow():
    try:
        from pygrabber.dshow_graph import FilterGraph
    except Exception:
        return []
    try:
        return FilterGraph().get_input_devices()
    except Exception:
        return []


def _camera_names_from_windows():
    if platform.system() != "Windows":
        return []

    command = [
        "powershell",
        "-NoProfile",
        "-Command",
        (
            "Get-CimInstance Win32_PnPEntity | "
            "Where-Object { $_.PNPClass -in @('Camera','Image') } | "
            "Select-Object -ExpandProperty Name"
        ),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5)
    except Exception:
        return []
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def list_cameras():
    names = _camera_names_from_directshow() or _camera_names_from_windows()
    return [{"index": index, "name": name} for index, name in enumerate(names)]


def camera_name(index):
    for camera in list_cameras():
        if camera["index"] == index:
            return camera["name"]
    return "Camera"


def choose_camera():
    cameras = list_cameras()
    ui.camera_menu(cameras)
    return ui.prompt_camera()
