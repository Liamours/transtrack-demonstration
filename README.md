# TransTrack Demonstration

Simple live fatigue-detection CLI using a PC-connected camera.

## Setup

```powershell
.\setup_env.ps1
conda activate transtrack_demo
```

To copy model files during setup, set the source repo first:

```powershell
$env:TRANSTRACK_API_ROOT="path\to\transtrack-api"
.\setup_env.ps1
```

Alternative:

```powershell
conda env create -f environment.yml
conda activate transtrack_demo
```

## Run

Use the new env, not another existing env:

```powershell
conda activate transtrack_demo
```

```powershell
python run_demo.py
```

Choose the camera index when prompted. Press `q` to stop.

To skip the prompt:

```powershell
python run_demo.py --camera 0
```

If a camera fails to open, try:

```powershell
python run_demo.py --camera 0 --backend msmf
```

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
