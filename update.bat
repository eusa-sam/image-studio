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
  echo Run start.bat once before updating.
  exit /b 1
)
".venv\Scripts\python.exe" -m scripts.update %*
exit /b %errorlevel%
