@echo off
REM ================================================================
REM  run_daily.bat - daily pipeline:
REM    data -> fundamentals (Mondays) -> factors -> scores -> signals (+ email) -> web page
REM  Run manually: double-click this file.  Schedule it: run setup_task.bat once.
REM  Logs: logs\daily_YYYYMMDD.log
REM ================================================================
setlocal

REM ---- settings ----
REM Score method: wf (walk-forward, recommended) / equal / ic / lgbm
set METHOD=wf
REM Download minute bars and compute minute factors: 1 = yes, 0 = no
set MINUTE=1
REM Push the web page snapshot to GitHub after each run: 1 = yes, 0 = no
set PUBLISH=0

cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set SCORE_METHOD=%METHOD%

set PY=python
if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe

if not exist logs mkdir logs
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"') do set TODAY=%%i
for /f %%i in ('powershell -NoProfile -Command "(Get-Date).DayOfWeek"') do set DOW=%%i
set LOG=logs\daily_%TODAY%.log

echo ===== %DATE% %TIME% start ===== >> "%LOG%"

set MFLAG=
set FFLAG=--no-minute
if "%MINUTE%"=="1" set MFLAG=--minute
if "%MINUTE%"=="1" set FFLAG=

"%PY%" data.py update %MFLAG% >> "%LOG%" 2>&1
if errorlevel 1 goto fail
REM Fundamentals change slowly: refresh once a week (only companies older than 7 days are re-downloaded)
REM A failure here is not fatal: factors fall back to the facts already downloaded
if "%DOW%"=="Monday" "%PY%" fundamentals.py update >> "%LOG%" 2>&1
"%PY%" factors.py %FFLAG% >> "%LOG%" 2>&1
if errorlevel 1 goto fail
"%PY%" model.py --method %METHOD% >> "%LOG%" 2>&1
if errorlevel 1 goto fail
"%PY%" daily_signal.py --method %METHOD% >> "%LOG%" 2>&1
if errorlevel 1 goto fail
"%PY%" export_site.py >> "%LOG%" 2>&1
if errorlevel 1 goto fail
if "%PUBLISH%"=="1" call publish.bat >> "%LOG%" 2>&1
if errorlevel 1 goto fail

echo ===== %DATE% %TIME% done ===== >> "%LOG%"
exit /b 0

:fail
echo ===== %DATE% %TIME% FAILED, see log above ===== >> "%LOG%"
"%PY%" daily_signal.py notify-error "%LOG%" >> "%LOG%" 2>&1
exit /b 1
