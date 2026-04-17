@echo off
REM REE experiment runner setup for Windows
REM Run once from the ree-v3 directory to create the venv and install dependencies.
REM Requires Python 3.10+ and git already installed.

echo === REE Windows Setup ===

REM Create venv in ree-v3\.venv
if exist ".venv" (
    echo [setup] .venv already exists -- skipping creation
) else (
    echo [setup] Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create venv. Is Python installed and on PATH?
        pause
        exit /b 1
    )
)

echo [setup] Upgrading pip...
.venv\Scripts\python -m pip install --upgrade pip --quiet

echo [setup] Installing numpy...
.venv\Scripts\pip install numpy --quiet

echo [setup] Installing PyTorch with CUDA 12.6 support (RTX 5070)...
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu126 --quiet

echo [setup] Verifying install...
.venv\Scripts\python -c "import torch, numpy; print('[setup] torch', torch.__version__, '/ CUDA available:', torch.cuda.is_available(), '/', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU'); print('[setup] numpy', numpy.__version__)"

if errorlevel 1 (
    echo [ERROR] Verification failed -- check output above.
    pause
    exit /b 1
)

echo.
echo === Setup complete ===
echo To start the runner:
echo   .venv\Scripts\python experiment_runner.py --auto-sync --loop --loop-interval 120 --machine EWIN-PC
echo.
pause
