@echo off
REM ================================================================
REM  setup_task.bat - register a Windows scheduled task "QuantDaily"
REM  Runs run_daily.bat Mon-Fri at 8:30 PM (computer local time).
REM  Wakes the computer if asleep; runs later if the computer was off.
REM  Run once (double-click). To remove: schtasks /Delete /TN QuantDaily /F
REM ================================================================
setlocal
set DIR=%~dp0
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$a = New-ScheduledTaskAction -Execute '%DIR%run_daily.bat' -WorkingDirectory '%DIR%';" ^
  "$t = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At 8:30PM;" ^
  "$s = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun -ExecutionTimeLimit (New-TimeSpan -Hours 6);" ^
  "Register-ScheduledTask -TaskName 'QuantDaily' -Action $a -Trigger $t -Settings $s -Force | Out-Null;" ^
  "Write-Host 'Scheduled task QuantDaily created: Mon-Fri 8:30 PM'"
if errorlevel 1 (
  echo Failed to create the task. Try right-click - Run as administrator.
) else (
  echo Test it now with:  schtasks /Run /TN QuantDaily
)
pause
