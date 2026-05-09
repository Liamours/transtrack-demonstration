param(
    [string]$EnvName = "transtrack_demo",
    [string]$PythonVersion = "3.11",
    [string]$SourceRoot = $env:TRANSTRACK_API_ROOT
)

Write-Host "Creating conda environment: $EnvName"
conda create -n $EnvName python=$PythonVersion -y
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Installing dependencies..."
conda run -n $EnvName pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Installing test dependencies..."
conda run -n $EnvName pip install -r requirements-dev.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($SourceRoot) {
    Write-Host "Copying model files..."
    New-Item -ItemType Directory -Force models\classifier, models\mediapipe | Out-Null
    Copy-Item -Force "$SourceRoot\models\classifier\best_val_f1.pth" "models\classifier\best_val_f1.pth"
    Copy-Item -Force "$SourceRoot\models\mediapipe\face_landmarker.task" "models\mediapipe\face_landmarker.task"
} else {
    Write-Host "Model copy skipped. Set TRANSTRACK_API_ROOT or pass -SourceRoot."
}

Write-Host ""
Write-Host "Done."
Write-Host "Activate with: conda activate $EnvName"
Write-Host "Run demo:      python run_demo.py"
