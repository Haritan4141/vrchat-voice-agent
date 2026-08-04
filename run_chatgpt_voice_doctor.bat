@echo off
setlocal
cd /d "%~dp0"
if not exist "config\chatgpt_voice.toml" copy "config\chatgpt_voice.example.toml" "config\chatgpt_voice.toml" >nul
uv sync --quiet
uv run chatgpt-voice-doctor --config config\chatgpt_voice.toml --live-seconds 8
echo.
pause
