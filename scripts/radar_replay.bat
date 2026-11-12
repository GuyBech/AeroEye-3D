@echo off
chcp 65001 >nul
title AeroEye-3D - Radar replay
cd /d "%~dp0..\vision"

echo ============================================
echo   AeroEye-3D  ^|  Radar replay (no hardware)
echo ============================================
echo.
echo   Plays a recorded session on the radar console.
echo   F = full screen   M = mute   Q = quit
echo.

"..\.venv\Scripts\python.exe" radar_replay.py

echo.
if errorlevel 1 (
  echo ------------------------------------------
  echo   STOPPED WITH AN ERROR - see above.
  echo ------------------------------------------
) else (
  echo   Stopped normally.
)
echo.
pause
