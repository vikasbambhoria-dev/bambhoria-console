@echo off
setlocal ENABLEDELAYEDEXPANSION

set "HOSTS=%SystemRoot%\System32\drivers\etc\hosts"
set "BACKUP=%SystemRoot%\System32\drivers\etc\hosts.bak"
set "ENTRY=127.0.0.1 bambhoriaquantum.in"

rem Check for administrative privileges
net session >nul 2>&1
if %errorLevel% NEQ 0 (
	echo Requesting administrator privileges to modify hosts file...
	powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -Verb RunAs -FilePath '%~f0'"
	exit /b
)

echo Backing up hosts file to "%BACKUP%"...
copy /Y "%HOSTS%" "%BACKUP%" >nul 2>&1

rem Check if the entry already exists
findstr /R /C:"^[ ]*127\.0\.0\.1[ ]\+bambhoriaquantum\.in[ ]*$" "%HOSTS%" >nul 2>&1
if %errorLevel%==0 (
	echo Mapping already present in hosts: %ENTRY%
) else (
	echo Adding mapping to hosts: %ENTRY%
	>>"%HOSTS%" echo %ENTRY%
)

echo Flushing DNS cache...
ipconfig /flushdns >nul 2>&1

echo Done. You can now open http://bambhoriaquantum.in:5000/ in your browser.
endlocal
http://bambhoriaquantum.in.