@echo off
setlocal EnableExtensions

if not defined REPO_ROOT (
  for %%R in ("%~dp0..") do set "REPO_ROOT=%%~fR"
)
if not defined VOICE_CONTROL_PORT set "VOICE_CONTROL_PORT=18765"

call :is_running
if not errorlevel 1 (
  echo VRChat Voice Agent production monitor is already running.
  exit /b 0
)

echo Starting VRChat Voice Agent production monitor...
start "VRChat Voice Agent Monitor" /min "%ComSpec%" /d /c call "%REPO_ROOT%\controls\run_chatgpt_voice_production.bat"

for /l %%N in (1,1,45) do (
  call :is_running
  if not errorlevel 1 (
    echo VRChat Voice Agent production monitor is ready.
    exit /b 0
  )
  >nul "%SystemRoot%\System32\timeout.exe" /t 1 /nobreak
)

echo [ERROR] VRChat Voice Agent production monitor did not become ready.
echo Check the minimized monitor window for details.
exit /b 1

:is_running
"%SystemRoot%\System32\netstat.exe" -ano -p tcp | "%SystemRoot%\System32\findstr.exe" /R /C:":%VOICE_CONTROL_PORT% .*LISTENING" >nul
exit /b %errorlevel%
