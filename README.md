# TransTrack Demo

Local Streamlit demo for live fatigue inference.

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
