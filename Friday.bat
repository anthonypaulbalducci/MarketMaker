@echo off
REM ===================================================================
REM   friday.bat - Weekly close/predict/execute/publish workflow
REM
REM   Run after markets close on Fridays (4pm ET or later).
REM   Closes last week's positions, generates next week's picks,
REM   opens new positions, and publishes results to the website.
REM
REM   Drop this file into the same folder as simulation.py.
REM ===================================================================

setlocal
cd /d "%~dp0"

echo.
echo ====================================================
echo   STEP 1/5: Close all positions for this week
echo ====================================================
python simulation.py close
if errorlevel 1 goto error

echo.
echo ====================================================
echo   STEP 2/5: Generate predictions for next week
echo ====================================================
python simulation.py predict
if errorlevel 1 goto error

echo.
echo ====================================================
echo   STEP 3/5: Execute new positions
echo ====================================================
python simulation.py execute
if errorlevel 1 goto error

echo.
echo ====================================================
echo   STEP 4/5: Upload picks.json to S3
echo ====================================================
python generate_picks.py --s3
if errorlevel 1 goto error

echo.
echo ====================================================
echo   STEP 5/5: Upload performance chart to S3
echo ====================================================
python plot_performance.py --upload
if errorlevel 1 goto error

echo.
echo ====================================================
echo   All steps completed successfully.
echo ====================================================
echo.
echo   Hard-refresh https://preceptron.com/marketmaker.html
echo   (Ctrl+Shift+R) to see the updated picks and chart.
echo.
echo   If the site does not update immediately, the CloudFront
echo   edge cache may still be serving the old version. To force
echo   an invalidation, run:
echo.
echo     aws cloudfront create-invalidation --distribution-id YOUR_DIST_ID --paths "/picks.json" "/performance.png"
echo.
pause
exit /b 0

:error
echo.
echo ====================================================
echo   !!! A STEP FAILED - SEE OUTPUT ABOVE !!!
echo ====================================================
echo   Stopping. Fix the error before running again, or re-run
echo   the individual remaining steps manually.
echo.
pause
exit /b 1
