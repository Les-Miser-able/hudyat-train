@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\pythonw.exe" (
  start "" ".venv\Scripts\pythonw.exe" "%~dp0desktop.py"
  exit /b
) else (
  python desktop.py
)
if errorlevel 1 pause
