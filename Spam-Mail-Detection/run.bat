@echo off
REM ============================================================
REM  Spam Mail Detection System — Quick Setup & Run Script
REM  Run this after Python is installed
REM ============================================================

echo.
echo ========================================================
echo   Spam Mail Detection System - Setup ^& Run
echo ========================================================
echo.

REM -- Find Python
SET PYTHON_CMD=python
%PYTHON_CMD% --version >nul 2>&1
IF ERRORLEVEL 1 (
    SET PYTHON_CMD=py
    %PYTHON_CMD% --version >nul 2>&1
    IF ERRORLEVEL 1 (
        echo [ERROR] Python not found on PATH. Please restart your terminal
        echo         after installation completes, then run this script again.
        pause
        exit /b 1
    )
)

echo [1/3] Python found: 
%PYTHON_CMD% --version

echo.
echo [2/3] Installing dependencies...
%PYTHON_CMD% -m pip install --upgrade pip
%PYTHON_CMD% -m pip install -r requirements.txt

echo.
echo [3/3] Running training pipeline...
%PYTHON_CMD% main.py

echo.
echo ========================================================
echo   Training complete! Now launching Streamlit UI...
echo   Open browser at: http://localhost:8501
echo ========================================================
echo.
%PYTHON_CMD% -m streamlit run app.py
