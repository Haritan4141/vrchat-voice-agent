@echo off
setlocal
for %%R in ("%~dp0..") do set "REPO_ROOT=%%~fR"
cd /d "%REPO_ROOT%"

if not exist "%REPO_ROOT%\system_prompt.txt" goto :missing_prompt

echo ChatGPT desktop must be running in Codex mode and idle.
echo Opening a new Codex task, starting GPT Live, and applying the prompt...

call :find_uv
if errorlevel 1 goto :missing_uv
"%UV_EXE%" sync --quiet
if errorlevel 1 goto :failed
call "%REPO_ROOT%\controls\ensure_voice_monitor.bat"
if errorlevel 1 goto :monitor_failed
"%UV_EXE%" run chatgpt-voice-prompt --prompt-file "%REPO_ROOT%\system_prompt.txt" --start-voice --require-codex --wait-seconds 30 --voice-wait-seconds 45 --voice-stabilization-seconds 12
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
echo [ERROR] system_prompt.txt was not found in the repository root.
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

:monitor_failed
echo.
echo [ERROR] Production monitoring could not be started.
echo The thinking indicator will not work until the monitor is running.
pause
exit /b 1
