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

REM torch is PINNED on purpose. machine_class() (experiments/_lib/arm_fingerprint.py)
REM includes the torch version, and arm-reuse only matches within one class -- so an
REM unpinned install silently splits a machine into its own class whenever it is
REM rebuilt on a different day. That is exactly how the cloud fleet ended up on three
REM different builds on 2026-07-19. See ree-v3/requirements.txt for the full note.
REM This box keeps the CUDA build (it has a GPU); the cloud fleet uses +cpu.
echo [setup] Installing PyTorch 2.12.0 with CUDA 12.6 support (RTX 5070)...
.venv\Scripts\pip install "torch==2.12.0" --index-url https://download.pytorch.org/whl/cu126 --quiet

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
