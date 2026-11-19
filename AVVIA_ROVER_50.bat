@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (echo Ambiente Python non trovato. & pause & exit /b 1)
".venv\Scripts\python.exe" avvia_rover_50.py
if errorlevel 1 pause
