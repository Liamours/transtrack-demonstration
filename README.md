# TransTrack Demo

Simple camera demo for live fatigue inference.

## Setup

```powershell
$env:TRANSTRACK_API_ROOT="path\to\transtrack-api"
.\setup_env.ps1
conda activate transtrack_demo
```

## Run

```powershell
python run_demo.py
```

Optional:

```powershell
python run_demo.py --camera 0
python run_demo.py --camera 0 --backend msmf
```

Press `q` to stop. `eyes_closed` and `yawning` are shown as warnings.

## Test

```powershell
pip install -r requirements-dev.txt
python -m pytest -q
```
