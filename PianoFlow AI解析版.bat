@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHON=%~dp0.venv-pianoflow\Scripts\python.exe"
if not exist "%PYTHON%" (
  echo PianoFlow Python環境が見つかりません。
  pause
  exit /b 1
)
start "" "http://127.0.0.1:4176/PianoFlow_AI.html"
"%PYTHON%" pianoflow_server.py --port 4176
