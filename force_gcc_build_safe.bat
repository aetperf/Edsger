@echo off
REM Safe GCC build script for older MinGW versions
REM Disables problematic LTO for compatibility

echo ===================================================================
echo SAFE GCC BUILD FOR EDSGER ON WINDOWS (No LTO)
echo ===================================================================
echo.

REM Check GCC availability
gcc --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] GCC not found! Please install MinGW-w64 first.
    pause
    exit /b 1
)

echo [INFO] GCC found, using safe flags for older MinGW...
gcc --version

REM Clean previous builds
echo [INFO] Cleaning previous builds...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
for /d %%i in (*.egg-info) do rmdir /s /q "%%i"
del /q src\edsger\*.c 2>nul
del /q src\edsger\*.pyd 2>nul

REM Set environment for older GCC (no LTO)
echo [INFO] Setting safe GCC environment (no LTO)...
set CC=gcc
set CXX=g++
set CFLAGS=-O3 -ffast-math -funroll-loops
set CXXFLAGS=-O3 -ffast-math -funroll-loops
set LDFLAGS=

echo [INFO] CC=%CC%
echo [INFO] CFLAGS=%CFLAGS%

REM Build with safe flags
echo [INFO] Building with safe GCC flags...
python setup.py build_ext --compiler=mingw32 --inplace --force

if errorlevel 1 (
    echo [ERROR] Build failed! Trying basic optimization...
    set CFLAGS=-O2
    set CXXFLAGS=-O2
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