$EnvName = "transtrack_demo"
$PythonVersion = "3.11"

Write-Host "Creating conda environment: $EnvName"
conda create -n $EnvName python=$PythonVersion -y
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Installing dependencies..."
conda run -n $EnvName pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Installing test dependencies..."
conda run -n $EnvName pip install -r requirements-dev.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "Done."
Write-Host "Activate with: conda activate $EnvName"
Write-Host "Run demo:      python run_demo.py"
