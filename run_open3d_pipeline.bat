@echo off
setlocal
cd /d "%~dp0"

set PYTHON_EXE=C:\Users\pitsi\AppData\Local\Programs\Python\Python311\python.exe
if not exist "%PYTHON_EXE%" set PYTHON_EXE=python

echo =======================================================
echo    OPEN3D SCAN RECONSTRUCTION PIPELINE
echo =======================================================
echo.
echo  [1] ICP + Screened Poisson  (works on all sessions)
echo  [2] TSDF Volumetric Fusion  (AI voting, needs RGB+depth)
echo  [3] Both pipelines
echo.
set /p CHOICE="Select mode (1/2/3) [default=1]: "
if "%CHOICE%"=="" set CHOICE=1

if "%CHOICE%"=="1" (
    set MODE=icp
    echo [ICP+Poisson mode selected]
) else if "%CHOICE%"=="2" (
    set MODE=tsdf
    echo [TSDF Volumetric Fusion mode selected]
) else if "%CHOICE%"=="3" (
    set MODE=both
    echo [Both pipelines selected]
) else (
    set MODE=icp
    echo [Invalid input - defaulting to ICP+Poisson]
)

echo.
echo Starting Open3D pipeline in %MODE% mode...
echo.
"%PYTHON_EXE%" tools\open3d_pipeline.py --session scans\session_stages --mode %MODE%

echo.
echo =======================================================
echo  Done! Check scans\ for output files.
echo =======================================================
pause
