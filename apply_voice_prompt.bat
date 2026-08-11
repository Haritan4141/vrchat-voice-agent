@echo off
setlocal
cd /d "%~dp0"

if not exist "%~dp0system_prompt.txt" goto :missing_prompt

echo ChatGPT desktop must be running and idle.
echo Opening a new task, starting GPT Live, and applying the prompt...

call :find_uv
if errorlevel 1 goto :missing_uv
"%UV_EXE%" sync --quiet
if errorlevel 1 goto :failed
"%UV_EXE%" run chatgpt-voice-prompt --prompt-file "%~dp0system_prompt.txt" --start-voice --wait-seconds 30 --voice-wait-seconds 45
set "RESULT=%errorlevel%"
echo.
if "%RESULT%"=="0" (
  echo Run this file again whenever you need to reapply the prompt.
)
pause
exit /b %RESULT%

:find_uv
for %%I in (uv.exe) do if not "%%~$PATH:I"=="" set "UV_EXE=%%~$PATH:I"
if defined UV_EXE exit /b 0
if exist "%LOCALAPPDATA%\Microsoft\WinGet\Links\uv.exe" set "UV_EXE=%LOCALAPPDATA%\Microsoft\WinGet\Links\uv.exe"
if defined UV_EXE exit /b 0
if exist "%USERPROFILE%\.local\bin\uv.exe" set "UV_EXE=%USERPROFILE%\.local\bin\uv.exe"
if defined UV_EXE exit /b 0
exit /b 1

:missing_prompt
echo [ERROR] system_prompt.txt was not found.
echo Place it in the same folder as this batch file.
echo.
pause
exit /b 1

:missing_uv
echo [ERROR] uv was not found.
echo Run this command in PowerShell, then try again:
echo   winget install --id=astral-sh.uv -e
echo.
pause
exit /b 1

:failed
echo.
echo [ERROR] Dependency setup failed.
pause
exit /b 1
