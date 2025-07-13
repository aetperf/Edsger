@echo off
REM Force GCC build script for Edsger on Windows
REM This script bypasses setuptools compiler detection issues

echo ===================================================================
echo FORCE GCC BUILD FOR EDSGER ON WINDOWS
echo ===================================================================
echo.

REM Check GCC availability
gcc --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] GCC not found! Please install MinGW-w64 first.
    echo   conda install -c conda-forge m2w64-toolchain
    pause
    exit /b 1
)

echo [INFO] GCC found, proceeding with forced GCC build...
gcc --version

REM Clean previous builds
echo [INFO] Cleaning previous builds...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
for /d %%i in (*.egg-info) do rmdir /s /q "%%i"
del /q src\edsger\*.c 2>nul
del /q src\edsger\*.pyd 2>nul

REM Set environment to force GCC usage
echo [INFO] Setting environment to force GCC usage...
set CC=gcc
set CXX=g++
set CFLAGS=-Ofast -flto -march=native -ffast-math -funroll-loops
set CXXFLAGS=-Ofast -flto -march=native -ffast-math -funroll-loops
set LDFLAGS=-flto

REM Disable MSVC detection
set DISTUTILS_USE_SDK=1
set MSSdk=1

echo [INFO] CC=%CC%
echo [INFO] CXX=%CXX%
echo [INFO] CFLAGS=%CFLAGS%

REM Force setuptools to use Unix compiler on Windows
echo [INFO] Building with forced GCC compiler...
python -c "import distutils.util; distutils.util.get_platform = lambda: 'linux-x86_64'"
python setup.py build_ext --compiler=unix --inplace --force

if errorlevel 1 (
    echo [ERROR] Build failed!
    echo.
    echo Trying alternative method...
    python setup.py build_ext --compiler=mingw32 --inplace --force
)

echo [INFO] Installing in development mode...
pip install -e . --force-reinstall --no-deps

echo.
echo [INFO] Verifying installation...
python -c "import edsger; print(f'Edsger version: {edsger.__version__}')"

echo.
echo ===================================================================
echo BUILD COMPLETED - Test performance with:
echo   cd scripts
echo   python benchmark_comparison_os.py -r 3
echo ===================================================================
pause