@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_CMD="

if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON_CMD=%~dp0.venv\Scripts\python.exe"
) else (
    where py >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_CMD=py -3"
    ) else (
        where python >nul 2>nul
        if not errorlevel 1 (
            set "PYTHON_CMD=python"
        )
    )
)

if not defined PYTHON_CMD (
    echo Python was not found.
    echo Install Python 3.10+ or create .venv first.
    pause
    exit /b 1
)

%PYTHON_CMD% -m vrchat_ai_tool gui %*
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo VRChat AI Tool GUI exited with code %EXIT_CODE%.
    pause
)

exit /b %EXIT_CODE%
