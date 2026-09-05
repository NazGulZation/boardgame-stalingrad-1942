@echo off
rem Stalingrad 1942 - start the game server and open the browser
cd /d "%~dp0"
start "Stalingrad 1942 server" "C:\Anaconda\python.exe" app.py
timeout /t 2 >nul
start "" http://127.0.0.1:5000
echo Server is running in its own window. Close that window to stop the game.
