@echo off
setlocal
cd /d "%~dp0"
echo Starting VRChat Voice Agent production monitor.
echo This process keeps running until you press Ctrl+C or close this window.
echo The diagnostic logger is not required for production use.
echo.
call "%~dp0launch_voice_control.bat"
exit /b %errorlevel%
