@echo off
setlocal
for %%R in ("%~dp0..") do set "REPO_ROOT=%%~fR"
cd /d "%REPO_ROOT%"
call :find_uv
if errorlevel 1 goto :missing_uv
if not exist "config\chatgpt_voice.toml" copy "config\chatgpt_voice.example.toml" "config\chatgpt_voice.toml" >nul
"%UV_EXE%" sync --quiet
if errorlevel 1 goto :failed
"%UV_EXE%" run vrchat-voice-control --config config\chatgpt_voice.toml
set "RESULT=%errorlevel%"
echo.
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

:missing_uv
echo [ERROR] uv is not installed or could not be found.
echo Install it from PowerShell, then run this file again:
echo   winget install --id=astral-sh.uv -e
echo.
pause
exit /b 1

:failed
echo.
echo [ERROR] Dependency setup failed.
pause
exit /b 1
