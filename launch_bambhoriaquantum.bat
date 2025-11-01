@echo off
title Bambhoria Quantum Trading Platform - bambhoriaquantum.in
echo.
echo ===============================================
echo    BAMBHORIA QUANTUM TRADING PLATFORM
echo    Domain: bambhoriaquantum.in
echo    Ultimate AI Trading with Zerodha Integration
echo ===============================================
echo.
echo Starting Bambhoria Quantum Web Application...
echo Domain: bambhoriaquantum.in
echo Access: http://bambhoriaquantum.in
echo HTTPS: https://bambhoriaquantum.in
echo.

cd /d "d:\bambhoria\godeye_v50_plus_auto_full_do_best"

echo Setting environment variables...
set FLASK_APP=bambhoria_quantum_web_app.py
set FLASK_ENV=production
set DOMAIN=bambhoriaquantum.in

echo Starting Flask server...
python bambhoria_quantum_web_app.py

pause
