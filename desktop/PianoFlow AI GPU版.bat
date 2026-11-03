@echo off
chcp 65001 >nul
cd /d "%~dp0"

set "APP_PYTHON=%~dp0.venv-pianoflow\Scripts\python.exe"
set "GPU_PYTHON=%USERPROFILE%\anaconda3\envs\rf-denoise-titan\python.exe"
set "GPU_MODEL_ROOT=%~dp0models\learning-v431-gpu"

if not exist "%APP_PYTHON%" (
  echo PianoFlow Python environment is missing.
  pause
  exit /b 1
)
if not exist "%GPU_PYTHON%" (
  echo GTX TITAN Python environment is missing:
  echo %GPU_PYTHON%
  pause
  exit /b 1
)

echo [1/4] Checking GTX TITAN CUDA...
"%GPU_PYTHON%" "%~dp0pianoflow_gpu_worker.py" --self-test
if errorlevel 1 (
  echo CUDA check failed. GPU version was not started.
  pause
  exit /b 1
)

echo [2/4] Exporting existing trained weights for PyTorch 1.3 (no retraining)...
"%APP_PYTHON%" "%~dp0export_learning_v431_gpu_checkpoints.py" --output "%GPU_MODEL_ROOT%"
if errorlevel 1 (
  echo Model export failed. Check free disk space and model files.
  pause
  exit /b 1
)

echo [3/4] Testing all three models on CUDA...
"%GPU_PYTHON%" "%~dp0pianoflow_gpu_worker.py" --check-models --model-dir "%GPU_MODEL_ROOT%"
if errorlevel 1 (
  echo CUDA model test failed. GPU version was not started.
  pause
  exit /b 1
)

set "PIANOFLOW_LEARNING_GPU_WORKER=%GPU_PYTHON%"
set "PIANOFLOW_LEARNING_GPU_MODEL_ROOT=%GPU_MODEL_ROOT%"
set "PIANOFLOW_LEARNING_DEVICE=cuda"
set "PIANOFLOW_GPU_READY=1"

echo [4/4] Starting the GPU server and opening the browser...
"%APP_PYTHON%" "%~dp0pianoflow_gpu_launcher.py" --port 4177
if errorlevel 1 pause
