import platform
import subprocess


def list_camera_names():
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


def choose_camera():
    names = list_camera_names()

    print("\nTransTrack Live Fatigue Detection")
    print("---------------------------------")
    if names:
        print("Detected cameras:")
        for index, name in enumerate(names):
            print(f"  [{index}] {name}")
    else:
        print("No camera names detected. Try index 0 first.")

    raw = input("\nCamera index [0]: ").strip()
    return int(raw) if raw else 0
