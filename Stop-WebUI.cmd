@echo off
cd /d "%~dp0"
"vendor\GPT-SoVITS\.venv\Scripts\python.exe" "scripts\stop_webui.py"
pause
