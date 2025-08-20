@echo off
cls
title RTX 4090 Voice Assistant
echo ============================================================
echo    RTX 4090 VOICE ASSISTANT
echo    GPU Accelerated - Human Voice - 100%% Offline
echo ============================================================
echo.
echo Features:
echo - NVIDIA RTX 4090 GPU acceleration
echo - Human-like voice with Piper TTS
echo - Real-time speech recognition
echo - Context-aware conversations
echo.
echo Starting...
echo.

python main.py

if errorlevel 1 (
    echo.
    echo ============================================================
    echo [ERROR] Application failed to start
    echo.
    echo Please check:
    echo 1. Python is installed: python --version
    echo 2. Dependencies installed: pip install -r requirements.txt
    echo 3. Models downloaded: python download_model.py
    echo 4. Voices downloaded: python download_best_voices.py
    echo ============================================================
)

echo.
pause