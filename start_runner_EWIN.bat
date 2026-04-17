@echo off
REM Start the REE experiment runner on EWIN-PC.
REM Run from the ree-v3 directory. Run setup_windows.bat first if .venv is missing.

if not exist ".venv" (
    echo [ERROR] .venv not found. Run setup_windows.bat first.
    pause
    exit /b 1
)

echo [runner] Starting REE experiment runner on EWIN-PC...
.venv\Scripts\python experiment_runner.py --auto-sync --loop --loop-interval 120 --machine EWIN-PC
