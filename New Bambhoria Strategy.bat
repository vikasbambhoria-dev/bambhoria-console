@echo off
REM Wrapper to run the PowerShell scaffold script. Usage:
REM "New Bambhoria Strategy.bat" [-Force]
powershell -ExecutionPolicy Bypass -File "%~dp0scripts\new_bambhoria_strategy.ps1" %*
