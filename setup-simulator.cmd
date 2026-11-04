@echo off
cd /d "%~dp0"
where node >nul 2>nul
if errorlevel 1 (
  echo Install Node.js 24, then run this file again.
  exit /b 1
)
call npm.cmd ci --prefix simulator --no-audit --no-fund
if errorlevel 1 exit /b 1
echo HUDYAT browser simulator ready. Open Test models, then Simulate HUDYAT.
