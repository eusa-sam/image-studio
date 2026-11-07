@echo off
setlocal
cd /d "%~dp0"
set "UV_PYTHON_INSTALL_DIR=%CD%\data\cache\python"
set "UV_CACHE_DIR=%CD%\data\cache\uv"
set "TEMP=%CD%\data\cache\tmp"
set "TMP=%TEMP%"
if not exist "%TEMP%" mkdir "%TEMP%"
where uv >nul 2>&1
if errorlevel 1 (
  echo Install uv first: https://docs.astral.sh/uv/
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  uv sync --locked
  if errorlevel 1 exit /b 1
)
uv run --no-sync python -m scripts.server start %*
exit /b %errorlevel%
