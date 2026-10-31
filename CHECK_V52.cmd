@echo off
setlocal
cd /d "%~dp0"
if defined BIOCHAR_PYTHON (
 set "PYTHON=%BIOCHAR_PYTHON%"
) else (
 if exist ".venv\Scripts\python.exe" (
  set "PYTHON=.venv\Scripts\python.exe"
 ) else (
  set "PYTHON=python"
 )
)
"%PYTHON%" -m pytest tests\test_v52_apple_series.py -q
if errorlevel 1 exit /b 1
echo V52 source and integrity tests PASSED.
