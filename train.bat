@echo off
REM ============================================================
REM  Splatfacto training launcher - forces VS 2022 compiler
REM ============================================================

REM Save original directory
set "VSCMD_START_DIR=%CD%"

REM Initialize VS 2022 Build Tools environment FIRST (clean shell, no VS 2025 yet)
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64

REM Force the compiler to VS 2022 cl.exe explicitly
set "CC=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe"
set "CXX=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe"

REM Verify correct compiler
echo ---- Compiler check ----
"%CC%" 2>&1 | findstr /C:"Version"
echo ------------------------

cd /d "C:\Users\arthu\Documents\Estudos\UFSC\9FASE\PFC\DreamCatalyst-PFC"
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
set "MAX_JOBS=4"

"C:\Users\arthu\miniconda3\envs\3d_edit\Scripts\ns-train.exe" splatfacto --max-num-iterations 500 --vis tensorboard nerfstudio-data --data data/chair_processed --downscale-factor 4 --load-3D-points False
pause
