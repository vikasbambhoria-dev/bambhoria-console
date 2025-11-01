@echo off
setlocal
REM Windows launcher for creating a new Bambhoria strategy scaffold

set SCRIPT_DIR=%~dp0
powershell -ExecutionPolicy Bypass -File "%SCRIPT_DIR%tools\new_bambhoria_strategy.ps1" %*
