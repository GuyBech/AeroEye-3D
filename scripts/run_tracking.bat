@echo off
chcp 65001 >nul
title AeroEye-3D - Tracking
cd /d "%~dp0..\vision"

echo ============================================
echo   AeroEye-3D  ^|  Laser Tracking System
echo ============================================
echo.
echo   Camera window will open in a few seconds.
echo   Close the window (or press Q) to stop.
echo.

"..\.venv\Scripts\python.exe" track_motion_3d.py COM4 1

echo.
if errorlevel 1 (
  echo ------------------------------------------
  echo   STOPPED WITH AN ERROR - see above.
  echo   Most common cause: another Python window
  echo   is still open and holding the camera or
  echo   COM4. Close it and run this again.
  echo ------------------------------------------
) else (
  echo   Stopped normally.
)
echo.
pause
