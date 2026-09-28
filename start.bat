@echo off
REM Start FX Hermes Trader and open it in your browser. Runs on this computer only (127.0.0.1).
setlocal EnableExtensions
set "PYTHONUTF8=1"
cd /d "%~dp0"
title FX Hermes Trader
if not exist ".venv\Scripts\python.exe" (
  echo  Not installed yet - running install.bat first...
  call "%~dp0install.bat"
  exit /b
)
set "URL=http://127.0.0.1:8503"
powershell -NoProfile -Command "try{(New-Object Net.Sockets.TcpClient('127.0.0.1',8503)).Close();exit 0}catch{exit 1}" >nul 2>nul
if not errorlevel 1 (
  echo  FX Hermes Trader is already running - opening %URL%
  start "" "%URL%"
  exit /b 0
)
if not defined NO_BROWSER start "" /b powershell -NoProfile -WindowStyle Hidden -Command "for($i=0;$i -lt 240;$i++){try{(New-Object Net.Sockets.TcpClient('127.0.0.1',8503)).Close();Start-Process '%URL%';exit}catch{Start-Sleep -Milliseconds 500}}"
echo.
echo  FX Hermes Trader is starting at %URL%
echo  Your browser opens by itself when it is ready. Close this window to stop.
echo.
.venv\Scripts\python.exe -m streamlit run ui\streamlit_app.py --server.port 8503 --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false
echo.
echo  FX Hermes Trader stopped.
if not defined NO_BROWSER pause
