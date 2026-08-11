@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul

if not exist "%~dp0system_prompt.txt" goto :missing_prompt

echo ChatGPTで新しいGPT Live音声タスクを開始してください。
echo 音声タスクが表示された状態になったら、この画面に戻ってください。
echo.
pause

call :find_uv
if errorlevel 1 goto :missing_uv
"%UV_EXE%" sync --quiet
if errorlevel 1 goto :failed
"%UV_EXE%" run chatgpt-voice-prompt --prompt-file "%~dp0system_prompt.txt" --wait-seconds 15
set "RESULT=%errorlevel%"
echo.
if "%RESULT%"=="0" (
  echo 必要になったときは、このファイルをもう一度実行すると指示を再適用できます。
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
echo [ERROR] system_prompt.txt が見つかりません。
echo このbatファイルと同じフォルダーに配置してください。
echo.
pause
exit /b 1

:missing_uv
echo [ERROR] uvが見つかりません。
echo PowerShellで次を実行してから、もう一度試してください:
echo   winget install --id=astral-sh.uv -e
echo.
pause
exit /b 1

:failed
echo.
echo [ERROR] 依存関係の準備に失敗しました。
pause
exit /b 1
