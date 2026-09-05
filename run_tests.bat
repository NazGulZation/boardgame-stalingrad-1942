@echo off
cd /d "%~dp0"
"C:\Anaconda\python.exe" -m unittest discover -s . -p "test_*.py" -v
echo.
pause
