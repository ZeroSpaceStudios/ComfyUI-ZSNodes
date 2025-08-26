@echo off
echo ============================================================
echo ComfyUI-ZSNodes Installation for Windows
echo ============================================================
echo.

REM Change to the directory where this batch file is located
cd /d "%~dp0"

REM Try to find python_embeded in various locations
set PYTHON_EXE=
if exist "..\..\..\python_embeded\python.exe" (
    set PYTHON_EXE=..\..\..\python_embeded\python.exe
    echo Found ComfyUI portable Python: %PYTHON_EXE%
) else if exist "..\..\python_embeded\python.exe" (
    set PYTHON_EXE=..\..\python_embeded\python.exe
    echo Found ComfyUI portable Python: %PYTHON_EXE%
) else (
    echo Could not find python_embeded directory.
    echo Trying system Python...
    set PYTHON_EXE=python
)

echo.
echo Running installation script...
echo.

REM Run the Python installation script
"%PYTHON_EXE%" install.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo Installation completed successfully!
    echo Please restart ComfyUI to use the new nodes.
    echo ============================================================
) else (
    echo.
    echo ============================================================
    echo Installation failed!
    echo Please check the error messages above.
    echo ============================================================
)

echo.
pause