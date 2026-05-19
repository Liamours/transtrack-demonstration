# TransTrack Demo

Local Streamlit demo for live fatigue inference.

## Description

This application is a local simulation interface for fatigue detection using a PC-connected camera. The demo captures live video, processes recent camera frames through the fatigue detection pipeline, and displays the current driver state in real time.

Detected labels:

- `normal`
- `eyes_closed`
- `yawning`

`eyes_closed` and `yawning` are treated as fatigue warning states. When a warning state is detected, the app shows a warning status and plays an alarm sound.

## Application Flow

```text
+--------------------+
| User selects camera |
+---------+----------+
          |
          v
+--------------------+
| Streamlit captures |
| live camera frames |
+---------+----------+
          |
          v
+--------------------+
| Rolling video clip |
| latest 20 seconds  |
+---------+----------+
          |
          v
+---------------------------+
| MediaPipe face landmarks  |
| EAR, MAR, head pose       |
+-------------+-------------+
              |
              v
+---------------------------+
| MultiScaleTCN classifier  |
| labels + confidence       |
+-------------+-------------+
              |
              v
+---------------------------+
| Streamlit UI              |
| live frame, label, stats  |
| alarm for fatigue labels  |
+---------------------------+
```

## Video Details

```text
Camera input       : PC-connected camera
Model window       : 20 seconds
Model sampling     : 10 FPS
Model timesteps    : 200 frames/features
Inference interval : every 10 seconds
Output labels      : normal, eyes_closed, yawning
Warning labels     : eyes_closed, yawning
Log output         : CSV in logs/
```

Optional visual diagnostics:

- eye and mouth landmarks
- EAR and MAR values on the video
- zoomed landmark view

## Setup

```powershell
$env:TRANSTRACK_API_ROOT="path\to\transtrack-api"
.\setup_env.ps1
conda activate transtrack_demo
```

## Run

```powershell
streamlit run streamlit_app.py
```

Optional:

```powershell
python run_demo.py
python run_demo.py --camera 0
python run_demo.py --camera 0 --backend msmf
```

Streamlit shows camera feed, label scores, video details, warning alarm, and fatigue statistics.
`eyes_closed` and `yawning` are warning labels.

## Test

```powershell
pip install -r requirements-dev.txt
python -m pytest -q
```
