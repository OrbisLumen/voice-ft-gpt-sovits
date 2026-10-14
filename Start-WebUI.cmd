@echo off
cd /d "%~dp0"
"vendor\GPT-SoVITS\.venv\Scripts\python.exe" "scripts\tts_webui.py" --mode main
pause
