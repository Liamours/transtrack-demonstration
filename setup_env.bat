@echo off
setlocal

set ENV_NAME=transtrack_demo
set PYTHON_VERSION=3.11

echo Creating conda environment: %ENV_NAME%
call conda create -n %ENV_NAME% python=%PYTHON_VERSION% -y
if errorlevel 1 goto error

echo Installing dependencies...
call conda run -n %ENV_NAME% pip install -r requirements.txt
if errorlevel 1 goto error

echo Installing test dependencies...
call conda run -n %ENV_NAME% pip install -r requirements-dev.txt
if errorlevel 1 goto error

echo.
echo Done.
echo Activate with:
echo   conda activate %ENV_NAME%
echo Run demo:
echo   python run_demo.py
goto end

:error
echo Setup failed.
exit /b 1

:end
endlocal
