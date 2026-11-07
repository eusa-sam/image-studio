@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo No local environment. Nothing to stop.
  exit /b 0
)
".venv\Scripts\python.exe" -m scripts.server stop %*
exit /b %errorlevel%
