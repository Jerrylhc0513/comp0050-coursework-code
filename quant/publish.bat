@echo off
REM ================================================================
REM  publish.bat - push the site/ snapshot to GitHub.
REM  Streamlit Community Cloud redeploys the web page automatically.
REM  One-time setup is described in README.md (section: public web page).
REM ================================================================
setlocal
cd /d "%~dp0"
where git >nul 2>&1
if errorlevel 1 (
  echo git not found. Install Git for Windows from https://git-scm.com
  exit /b 1
)
git add site app.py requirements.txt
git diff --cached --quiet
if not errorlevel 1 (
  echo Nothing new to publish.
  exit /b 0
)
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd"') do set TODAY=%%i
git commit -m "Daily site update %TODAY%"
if errorlevel 1 exit /b 1
git push
if errorlevel 1 exit /b 1
echo Published.
exit /b 0
