@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
title Fly Abstraction Lab - RTX 5090 setup
echo Запуск безопасной проверки и настройки Docker для RTX 5090...
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup_friend_pc.ps1" -Action Guided
set "SETUP_EXIT=%ERRORLEVEL%"
if not "%SETUP_EXIT%"=="0" (
  echo.
  echo ОШИБКА: настройка остановлена. Прочитайте сообщение выше и запустите START_HERE.cmd снова.
) else (
  echo.
  echo Проверка завершена. Данные и результаты сохранены в папках data и results.
)
echo.
pause
exit /b %SETUP_EXIT%
