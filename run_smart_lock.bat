@echo off
echo Smart Lock System - Hand Gesture Password
echo ==========================================
echo.
echo Starting Smart Lock System...
echo.

REM 
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.7 or higher from https://python.org
    pause
    exit /b 1
)

REM 
python run_smart_lock.py

REM 
pause 