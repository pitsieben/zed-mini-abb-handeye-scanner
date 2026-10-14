@echo off
setlocal
cd /d "%~dp0"

set PYTHON_EXE=C:\Users\pitsi\AppData\Local\Programs\Python\Python311\python.exe

if not exist "%PYTHON_EXE%" (
    if exist "venv\Scripts\python.exe" (
        set PYTHON_EXE=venv\Scripts\python.exe
    ) else (
        set PYTHON_EXE=python
    )
)

echo =======================================================
echo    STARTING ZED MINI + ROBOT WEB SCANNER APP
echo =======================================================
echo.
echo Opening browser at http://localhost:8080 ...
start http://localhost:8080

"%PYTHON_EXE%" web_scanner_app.py

pause
