@echo off
set PYTHON_EXE=C:\Users\pitsi\AppData\Local\Programs\Python\Python311\python.exe

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python 3.11 not found at %PYTHON_EXE%
    pause
    exit /b 1
)

echo =======================================================
echo    RUNNING OPEN3D MULTI-WAY ICP AND SURFACE FUSION
echo =======================================================

"%PYTHON_EXE%" tools\open3d_pipeline.py --session scans\session_stages %*

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [+] Point cloud and mesh exported to 1:1 Millimeters for Rhino!
    echo     - scans\latest_workpiece_mm.ply
    echo     - scans\latest_workpiece_mesh_mm.obj
    echo.
) else (
    echo.
    echo [!] Pipeline failed. Check if scans\session_stages exists.
)
