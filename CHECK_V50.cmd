@echo off
setlocal
cd /d "%~dp0"
if defined BIOCHAR_PYTHON (
  set "PY=%BIOCHAR_PYTHON%"
) else (
  set "PY=python"
)
"%PY%" -m pytest tests\test_v50_model_utility.py -q
if errorlevel 1 (
 echo [FAIL] V50 model utility check failed.
 exit /b 1
)
echo [OK] V50 research test passed; deployed models were not changed.
endlocal
