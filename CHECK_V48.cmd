@echo off
setlocal
cd /d "%~dp0"
set "PYEXE=%BIOCHAR_PYTHON%"
if not defined PYEXE if exist ".venv\Scripts\python.exe" set "PYEXE=.venv\Scripts\python.exe"
if not defined PYEXE (
 echo Set BIOCHAR_PYTHON to the full path to python.exe in your .venv
 echo Or create .venv inside this project.
 exit /b 1
)
"%PYEXE%" -m scripts.verify_v48 --full-checksums
if errorlevel 1 exit /b 1
"%PYEXE%" -m pytest tests\test_v48_material_workflow.py tests\test_v44_workflow.py tests\test_v43_scenarios.py tests\test_v41_scope.py tests\test_v47_integrated.py -q
exit /b %ERRORLEVEL%
