@echo off
setlocal
cd /d "%~dp0"

set PYTHON_EXE=C:\Users\pitsi\AppData\Local\Programs\Python\Python311\python.exe
if not exist "%PYTHON_EXE%" set PYTHON_EXE=python

echo =======================================================
echo    3D GAUSSIAN SPLATTING DATASET EXPORTER
echo =======================================================
echo.
echo  Exports the current scan session to a dataset ready
echo  for 3DGS / NeRF training.
echo.
echo  Output: scans\3dgs_dataset\
echo    - images\            (RGB frames)
echo    - transforms.json    (nerfstudio / instant-NGP format)
echo    - sparse\0\          (COLMAP format for gaussian-splatting)
echo    - HOWTO_3DGS.md      (training instructions)
echo.
echo Starting export...
echo.
"%PYTHON_EXE%" tools\export_3dgs_dataset.py --session scans\session_stages --out scans\3dgs_dataset

echo.
echo =======================================================
echo  Export complete!
echo.
echo  To train with gaussian-splatting:
echo    python train.py -s scans\3dgs_dataset --model_path output\robot_scan
echo.
echo  See scans\3dgs_dataset\HOWTO_3DGS.md for all options.
echo =======================================================
pause
