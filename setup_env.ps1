$EnvName = "transtrack_demo"
$PythonVersion = "3.11"
$SourceRoot = "C:\Users\lulay\Desktop\Activities\transtrack\transtrack-api"

Write-Host "Creating conda environment: $EnvName"
conda create -n $EnvName python=$PythonVersion -y
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Installing dependencies..."
conda run -n $EnvName pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Installing test dependencies..."
conda run -n $EnvName pip install -r requirements-dev.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Copying model files..."
New-Item -ItemType Directory -Force models\classifier, models\mediapipe | Out-Null
Copy-Item -Force "$SourceRoot\models\classifier\best_val_f1.pth" "models\classifier\best_val_f1.pth"
Copy-Item -Force "$SourceRoot\models\mediapipe\face_landmarker.task" "models\mediapipe\face_landmarker.task"

Write-Host ""
Write-Host "Done."
Write-Host "Activate with: conda activate $EnvName"
Write-Host "Run demo:      python run_demo.py"
