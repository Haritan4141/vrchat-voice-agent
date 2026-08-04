@echo off
setlocal
cd /d "%~dp0"
if not exist "config\chatgpt_voice.toml" copy "config\chatgpt_voice.example.toml" "config\chatgpt_voice.toml" >nul
uv sync --quiet
uv run vrchat-voice-control --config config\chatgpt_voice.toml
echo.
pause
