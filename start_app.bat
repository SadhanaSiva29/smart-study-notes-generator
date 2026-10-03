@echo off
title Smart Study Notes Generator
echo ===================================================
echo     Launching Smart Study Notes Generator Web App
echo ===================================================
echo.
cd /d "%~dp0"
echo Starting Flask web server...
start http://127.0.0.1:5000
python web_app.py
pause
