@echo off
REM Train the sector rotation model with Optuna hyperparameter tuning.
REM
REM Pipeline (via run.py --tune):
REM   1. Download weekly sector data
REM   2. Preprocess features
REM   3. Hyperparameter tuning (Optuna)
REM   4. Train with best params
REM   5. Evaluate
REM
REM Usage:
REM   train.bat                                 (defaults: 30 trials)
REM   train.bat --tune-trials 100               (more trials)
REM   train.bat --tune-trials 50 --epochs 100   (extra args passed to run.py)

setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
) else (
    echo [warn] .venv\Scripts\activate.bat not found - using system Python
)

python run.py --tune %*
set EXITCODE=%ERRORLEVEL%

if not "%EXITCODE%"=="0" (
    echo.
    echo [error] Pipeline failed with exit code %EXITCODE%
)

pause
exit /b %EXITCODE%
