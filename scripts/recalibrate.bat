@echo off
chcp 65001 >nul
title AeroEye-3D - Recalibration
cd /d "%~dp0..\vision"

echo ============================================
echo   AeroEye-3D  ^|  Recalibration
echo ============================================
echo.
echo   ONLY needed if the turret or camera moved.
echo.
echo   HOW:  hold the ball still  -^>  press SPACE
echo         the turret centres the laser itself
echo         and records the point automatically.
echo         Then move the ball and repeat.
echo.
echo   Mouse click = mark the laser dot by hand
echo   X = skip this position   Q = done
echo.
echo   COVERAGE: the window shows a 3x3 grid with the
echo   number of points per cell AT THE BALL'S CURRENT
echo   DEPTH [far / mid / near]. Red 0 = still empty.
echo   Fill all 9 cells at all 3 depths = 27 cells.
echo.
echo   Red circle = ball touches the image edge:
echo   SPACE is ignored - move the ball inward.
echo.
pause

"..\.venv\Scripts\python.exe" calib_ball.py COM4 1

echo.
echo ============================================
echo   Fitting the model to the collected points
echo ============================================
"..\.venv\Scripts\python.exe" fit_ball_calib.py
echo.
pause
