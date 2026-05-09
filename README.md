# TransTrack Demonstration

Simple live fatigue-detection CLI using a PC-connected camera.

## Setup

```powershell
pip install -r requirements.txt
```

## Run

```powershell
python run_demo.py
```

Choose the camera index when prompted. Press `q` to stop.

The app:
- captures camera frames continuously
- runs the TransTrack MediaPipe + MultiScaleTCN pipeline
- displays the current label
- writes CSV logs to `logs/fatigue_inference.csv`

Default model path:

```text
models/classifier/best_val_f1.pth
```

## Test

```powershell
pip install -r requirements-dev.txt
python -m pytest -q
```
