@echo off
cls
echo ============================================================
echo    RTX 4090 VOICE ASSISTANT - SETUP
echo ============================================================
echo.

REM Check Python version
echo [1/5] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.9-3.11 from python.org
    pause
    exit /b 1
)
python --version
echo.

REM Check CUDA
echo [2/5] Checking GPU/CUDA...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [WARNING] NVIDIA GPU not detected or drivers not installed
    echo The assistant will work but run slower on CPU
) else (
    echo [OK] NVIDIA GPU detected
)
echo.

REM Install PyTorch with CUDA
echo [3/5] Installing PyTorch with CUDA support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
echo.

REM Install requirements
echo [4/5] Installing dependencies...
pip install -r requirements.txt
echo.

REM Download models
echo [5/5] Downloading AI models...
echo.
echo Downloading language model (Phi-2)...
python download_model.py
echo.
echo Downloading voice models (Ryan & Amy)...
python download_best_voices.py

echo.
echo ============================================================
echo    SETUP COMPLETE!
echo ============================================================
echo.
echo To start the assistant:
echo   python main.py
echo.
echo Or double-click:
echo   run.bat
echo.
pause