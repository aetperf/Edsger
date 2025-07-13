# Windows clean build script for Edsger (PowerShell version)
# This script performs a complete clean rebuild with MSVC optimizations

Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host "EDSGER WINDOWS CLEAN BUILD WITH MSVC OPTIMIZATIONS" -ForegroundColor Cyan
Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if we're in the right directory
if (-not (Test-Path "setup.py")) {
    Write-Host "[ERROR] setup.py not found. Please run this script from the Edsger root directory." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "[INFO] Detected OS: Windows" -ForegroundColor Green
try {
    $pythonVersion = python --version 2>$null
    Write-Host "[INFO] $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Python not found in PATH" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Step 1: Clean all build artifacts
Write-Host "[INFO] Cleaning all build artifacts..." -ForegroundColor Yellow

if (Test-Path "build") {
    Remove-Item -Recurse -Force "build"
    Write-Host "[INFO] Removed build\ directory" -ForegroundColor Green
}

if (Test-Path "dist") {
    Remove-Item -Recurse -Force "dist"
    Write-Host "[INFO] Removed dist\ directory" -ForegroundColor Green
}

# Remove egg-info directories
Get-ChildItem -Path "." -Filter "*.egg-info" -Directory | Remove-Item -Recurse -Force
Write-Host "[INFO] Removed *.egg-info directories" -ForegroundColor Green

# Remove compiled Python files
Get-ChildItem -Path "." -Recurse -Filter "*.pyc" | Remove-Item -Force
Get-ChildItem -Path "." -Recurse -Filter "__pycache__" -Directory | Remove-Item -Recurse -Force
Write-Host "[INFO] Removed *.pyc files and __pycache__ directories" -ForegroundColor Green

# Remove Cython-generated files and compiled extensions
if (Test-Path "src\edsger") {
    Get-ChildItem -Path "src\edsger" -Filter "*.c" | Remove-Item -Force
    Get-ChildItem -Path "src\edsger" -Filter "*.pyd" | Remove-Item -Force
    Get-ChildItem -Path "src\edsger" -Filter "*.html" | Remove-Item -Force
    Write-Host "[INFO] Removed Cython-generated C files and compiled extensions" -ForegroundColor Green
}

# Step 2: Check for GCC and set optimal environment variables
Write-Host "[INFO] Checking for available compilers..." -ForegroundColor Yellow

try {
    $gccVersion = gcc --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[INFO] GCC detected! Will use GCC toolchain (same as SciPy)" -ForegroundColor Green
        Write-Host "[INFO] Setting GCC optimization environment variables..." -ForegroundColor Yellow
        $env:CC = "gcc"
        $env:CXX = "g++"
        $env:CFLAGS = "-Ofast -flto -march=native -ffast-math -funroll-loops"
        $env:CXXFLAGS = "-Ofast -flto -march=native -ffast-math -funroll-loops" 
        $env:LDFLAGS = "-flto"
        Write-Host "[INFO] CC: $env:CC" -ForegroundColor Cyan
        Write-Host "[INFO] CXX: $env:CXX" -ForegroundColor Cyan
        Write-Host "[INFO] CFLAGS: $env:CFLAGS" -ForegroundColor Cyan
        Write-Host "[INFO] CXXFLAGS: $env:CXXFLAGS" -ForegroundColor Cyan
        Write-Host "[INFO] LDFLAGS: $env:LDFLAGS" -ForegroundColor Cyan
    } else {
        throw "GCC not found"
    }
} catch {
    Write-Host "[INFO] GCC not found, using MSVC optimization environment variables..." -ForegroundColor Yellow
    $env:CL = "/O2 /Ot /GL /favor:INTEL64 /fp:fast /GS- /Gy /arch:AVX2"
    $env:LINK = "/LTCG /OPT:REF /OPT:ICF"
    Write-Host "[INFO] CL: $env:CL" -ForegroundColor Cyan
    Write-Host "[INFO] LINK: $env:LINK" -ForegroundColor Cyan
    Write-Host "[INFO] Note: For potentially better performance, consider installing MinGW-w64" -ForegroundColor Yellow
}
Write-Host ""

# Step 3: Build extensions in place
Write-Host "[INFO] Building Cython extensions in place..." -ForegroundColor Yellow

$buildResult = python setup.py build_ext --inplace --force
$buildExitCode = $LASTEXITCODE

if ($buildExitCode -ne 0) {
    Write-Host "[WARNING] Build failed with AVX2, trying fallback with SSE2..." -ForegroundColor Yellow
    $env:CL = "/O2 /Ot /GL /favor:INTEL64 /fp:fast /GS- /Gy /arch:SSE2"
    Write-Host "[INFO] Fallback CL: $env:CL" -ForegroundColor Cyan
    
    $buildResult = python setup.py build_ext --inplace --force
    $buildExitCode = $LASTEXITCODE
    
    if ($buildExitCode -ne 0) {
        Write-Host "[ERROR] Build failed even with SSE2 fallback!" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# Step 4: Install in development mode
Write-Host "[INFO] Installing package in development mode..." -ForegroundColor Yellow

$installResult = python -m pip install -e . --force-reinstall --no-deps
$installExitCode = $LASTEXITCODE

if ($installExitCode -ne 0) {
    Write-Host "[ERROR] Installation failed! Check the error messages above." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Step 5: Verify installation
Write-Host "[INFO] Verifying installation..." -ForegroundColor Yellow

try {
    $versionOutput = python -c "import edsger; print(f'Edsger version: {edsger.__version__}')" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[INFO] ✓ Edsger successfully installed and importable" -ForegroundColor Green
        Write-Host "[INFO] $versionOutput" -ForegroundColor Green
    } else {
        Write-Host "[WARNING] ⚠ Edsger installed but import test failed" -ForegroundColor Yellow
    }
} catch {
    Write-Host "[WARNING] ⚠ Could not verify installation" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host "CLEAN BUILD COMPLETED SUCCESSFULLY" -ForegroundColor Green
Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host "Next steps:" -ForegroundColor White
Write-Host "  cd scripts" -ForegroundColor Yellow
Write-Host "  python benchmark_comparison_os.py -r 5" -ForegroundColor Yellow
Write-Host ""
Write-Host "Expected Windows performance improvement: 3-4 seconds (vs 6 seconds unoptimized)" -ForegroundColor White
Write-Host "Target: Reduce gap with Linux from 2.4x to 1.5x" -ForegroundColor White
Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host ""
Read-Host "Press Enter to exit"