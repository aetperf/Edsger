@echo off
REM Windows clean build script for Edsger
REM This script performs a complete clean rebuild with MSVC optimizations

echo ===================================================================
echo EDSGER WINDOWS CLEAN BUILD WITH MSVC OPTIMIZATIONS
echo ===================================================================
echo.

REM Check if we're in the right directory
if not exist setup.py (
    echo [ERROR] setup.py not found. Please run this script from the Edsger root directory.
    pause
    exit /b 1
)

echo [INFO] Detected OS: Windows
python --version 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found in PATH
    pause
    exit /b 1
)

REM Step 1: Clean all build artifacts
echo [INFO] Cleaning all build artifacts...

if exist build (
    rmdir /s /q build
    echo [INFO] Removed build\ directory
)

if exist dist (
    rmdir /s /q dist
    echo [INFO] Removed dist\ directory
)

REM Remove egg-info directories
for /d %%i in (*.egg-info) do (
    if exist "%%i" (
        rmdir /s /q "%%i"
        echo [INFO] Removed %%i directory
    )
)

REM Remove compiled Python files
for /r . %%i in (*.pyc) do del "%%i" 2>nul
for /d /r . %%i in (__pycache__) do rmdir /s /q "%%i" 2>nul
echo [INFO] Removed *.pyc files and __pycache__ directories

REM Remove Cython-generated files and compiled extensions
del /q src\edsger\*.c 2>nul
del /q src\edsger\*.pyd 2>nul
del /q src\edsger\*.html 2>nul
echo [INFO] Removed Cython-generated C files and compiled extensions

REM Step 2: Check for GCC and set optimal environment variables
echo [INFO] Checking for available compilers...

gcc --version >nul 2>&1
if errorlevel 1 (
    echo [INFO] GCC not found, using MSVC optimization environment variables...
    set "CL=/O2 /Ot /GL /favor:INTEL64 /fp:fast /GS- /Gy /arch:AVX2"
    set "LINK=/LTCG /OPT:REF /OPT:ICF"
    echo [INFO] CL: %CL%
    echo [INFO] LINK: %LINK%
    echo [INFO] Note: For potentially better performance, consider installing MinGW-w64
) else (
    echo [INFO] GCC detected! Will use GCC toolchain (same as SciPy^)
    echo [INFO] Setting GCC optimization environment variables...
    set "CFLAGS=-Ofast -flto -march=native -ffast-math -funroll-loops"
    set "CXXFLAGS=-Ofast -flto -march=native -ffast-math -funroll-loops"
    set "LDFLAGS=-flto"
    echo [INFO] CFLAGS: %CFLAGS%
    echo [INFO] CXXFLAGS: %CXXFLAGS%
    echo [INFO] LDFLAGS: %LDFLAGS%
)
echo.

REM Step 3: Build extensions in place
echo [INFO] Building Cython extensions in place...
python setup.py build_ext --inplace --force

if errorlevel 1 (
    echo [ERROR] Build failed! Check the error messages above.
    echo.
    echo Trying fallback with SSE2 instead of AVX2...
    set "CL=/O2 /Ot /GL /favor:INTEL64 /fp:fast /GS- /Gy /arch:SSE2"
    echo [INFO] Fallback CL: %CL%
    python setup.py build_ext --inplace --force
    
    if errorlevel 1 (
        echo [ERROR] Build failed even with SSE2 fallback!
        pause
        exit /b 1
    )
)

REM Step 4: Install in development mode
echo [INFO] Installing package in development mode...
python -m pip install -e . --force-reinstall --no-deps

if errorlevel 1 (
    echo [ERROR] Installation failed! Check the error messages above.
    pause
    exit /b 1
)

REM Step 5: Verify installation
echo [INFO] Verifying installation...
python -c "import edsger; print(f'Edsger version: {edsger.__version__}')" 2>nul

if errorlevel 1 (
    echo [WARNING] Edsger installed but import test failed
) else (
    echo [INFO] ✓ Edsger successfully installed and importable
)

echo.
echo ===================================================================
echo CLEAN BUILD COMPLETED SUCCESSFULLY
echo ===================================================================
echo Next steps:
echo   cd scripts
echo   python benchmark_comparison_os.py -r 5
echo.
echo Expected Windows performance improvement: 3-4 seconds (vs 6 seconds unoptimized)
echo Target: Reduce gap with Linux from 2.4x to 1.5x
echo ===================================================================
echo.
pause