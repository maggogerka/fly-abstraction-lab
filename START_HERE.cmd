@echo off
setlocal
cd /d "%~dp0"
title Fly Abstraction Lab - RTX 50-series setup
set "PS_EXE=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%PS_EXE%" goto no_powershell
echo Starting the safe RTX 50-series Docker setup...
"%PS_EXE%" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup_friend_pc.ps1" -Action Guided
set "SETUP_EXIT=%ERRORLEVEL%"
if not "%SETUP_EXIT%"=="0" goto failed
echo.
echo Setup finished. Data and results remain in the data and results folders.
goto finish

:no_powershell
set "SETUP_EXIT=1"
echo ERROR: Windows PowerShell 5.1 was not found at "%PS_EXE%".
goto finish

:failed
echo.
echo ERROR: setup stopped. Read the message above and run START_HERE.cmd again.

:finish
echo.
pause
exit /b %SETUP_EXIT%
