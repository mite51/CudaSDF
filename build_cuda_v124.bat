@echo off
setlocal

echo ===============================================================================
echo  Building CudaSDF with CUDA 12.4 (Compatibility Build)
echo ===============================================================================

REM Set Paths for CUDA 12.4
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
set "PATH=%CUDA_PATH%\bin;%PATH%"

echo.
echo [1/3] Cleaning previous build (build_v124)...
if exist build_v124 (
    rmdir /s /q build_v124
    if exist build_v124 (
        echo Error: Could not remove build_v124 directory. Is it open in another program?
        pause
        exit /b 1
    )
)
echo Cleaned.

echo.
echo [2/3] Configuring with CMake (Force CUDA 12.4)...
cmake -S . -B build_v124 -G "Visual Studio 17 2022" -T cuda="%CUDA_PATH%"
if %ERRORLEVEL% NEQ 0 (
    echo Error: CMake configuration failed.
    pause
    exit /b 1
)

echo.
echo [3/3] Building Release Configuration...
cmake --build build_v124 --config Release
if %ERRORLEVEL% NEQ 0 (
    echo Error: Build failed.
    pause
    exit /b 1
)

echo.
echo ===============================================================================
echo  SUCCESS!
echo  Executable is located at: build_v124\Release\CudaSDF.exe
echo ===============================================================================
pause

