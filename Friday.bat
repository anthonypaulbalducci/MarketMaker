@echo off
REM Friday weekly rotation: close last week's positions, get new picks,
REM execute new trades, then show portfolio status.
REM
REM Steps:
REM   1. simulation.py close    - close all open option positions
REM   2. simulation.py predict  - generate this week's sector picks
REM   3. simulation.py execute  - buy options for the new picks
REM   4. simulation.py status   - show resulting portfolio
REM
REM Per Instructions.txt the canonical schedule is close+predict on Friday and
REM execute on Monday morning. This script bundles all three for a same-day
REM rotation - comment out the execute step if you prefer to wait until Monday.

setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
) else (
    echo [warn] .venv\Scripts\activate.bat not found - using system Python
)

echo.
echo ====================================================================
echo STEP 1/4: Close existing positions
echo ====================================================================
python simulation.py close
if errorlevel 1 goto :fail

echo.
echo ====================================================================
echo STEP 2/4: Generate this week's picks
echo ====================================================================
python simulation.py predict
if errorlevel 1 goto :fail

echo.
echo ====================================================================
echo STEP 3/4: Execute new trades
echo ====================================================================
python simulation.py execute
if errorlevel 1 goto :fail

echo.
echo ====================================================================
echo STEP 4/4: Portfolio status
echo ====================================================================
python simulation.py status
if errorlevel 1 goto :fail

echo.
echo [ok] Friday rotation complete.
pause
exit /b 0

:fail
set EXITCODE=%ERRORLEVEL%
echo.
echo [error] Friday rotation failed at exit code %EXITCODE%
pause
exit /b %EXITCODE%
