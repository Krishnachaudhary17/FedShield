@echo off
title FedShield Launcher
echo =======================================================
echo          Launching FedShield Application
echo =======================================================
echo.

echo [1/2] Starting Flask API Server on port 5000...
start "FedShield API Server" cmd /k "cd /d %~dp0 && python api_server.py"

timeout /t 2 /nobreak >nul

echo [2/2] Starting Web UI Server on port 8080...
start "FedShield Web UI" cmd /k "cd /d %~dp0\FedShield_UI && python -m http.server 8080"

timeout /t 1 /nobreak >nul

echo Opening browser at http://localhost:8080...
start http://localhost:8080/index.html

echo.
echo Application started successfully!
echo - Dashboard: http://localhost:8080/index.html
echo - Fraud Analysis: http://localhost:8080/fraud_detection.html
echo - Backend API: http://127.0.0.1:5000
echo.
pause
