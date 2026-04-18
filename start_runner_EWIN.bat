@echo off
REM Start the REE experiment runner on EWIN-PC.
REM Run from the ree-v3 directory.
REM
REM Uses system python (not a venv). Requires: python, torch, numpy on PATH.

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] python not found on PATH. Install Python 3.10+ and retry.
    pause
    exit /b 1
)

echo [runner] Starting REE experiment runner on EWIN-PC...
python experiment_runner.py --auto-sync --loop --loop-interval 120 --machine EWIN-PC
