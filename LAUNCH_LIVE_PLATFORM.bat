@echo off
setlocal ENABLEDELAYEDEXPANSION

REM ==============================================
REM  BAMBHORIA QUANTUM - LIVE PLATFORM LAUNCHER
REM ==============================================

TITLE Bambhoria Quantum - Live Trading Platform
ECHO.
ECHO ==============================================
ECHO   BAMBHORIA QUANTUM - LIVE TRADING PLATFORM
ECHO   Serving via WSGI (waitress) on port 5000
ECHO ==============================================
ECHO.

REM Change to project directory
CD /D "%~dp0"

REM Activate virtual environment if present
IF EXIST .venv\Scripts\activate.bat (
    ECHO Activating virtual environment...
    CALL .venv\Scripts\activate.bat
) ELSE (
    ECHO No .venv found. Using system Python.
)

REM Ensure required packages are installed (quiet)
ECHO Installing minimal runtime dependencies (Flask, requests, dotenv, waitress)...
REM Avoid compiling heavy packages like pandas/numpy on Python 3.13
pip install -q "flask==2.3.3" "requests==2.31.0" "python-dotenv==1.0.0" "waitress==3.0.0"

REM Set environment
SET FLASK_ENV=production
SET PORT=5000

REM Start server with waitress
ECHO Starting server on http://127.0.0.1:5000 ...
start "Bambhoria Quantum Server" cmd /c waitress-serve --listen=127.0.0.1:5000 wsgi:application

REM Give server a moment to start
powershell -Command "Start-Sleep -Seconds 2"

REM Open browser
ECHO Opening browser...
start http://127.0.0.1:5000/

ECHO.
ECHO Live platform launched. Press any key to close this window.
PAUSE >NUL
endlocal
