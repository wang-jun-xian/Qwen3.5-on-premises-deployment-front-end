@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PY=python"
if exist "E:\conda\condaenvs\test_env\python.exe" set "PY=E:\conda\condaenvs\test_env\python.exe"
echo Starting Qwen3.5-4B Multimodal Assistant...
"%PY%" app.py --preload --port 7860
pause
