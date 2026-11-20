@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Create the project virtual environment and install requirements.txt first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" desktop_assistant.py
if errorlevel 1 pause
