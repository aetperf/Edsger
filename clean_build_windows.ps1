# Simple Windows build script for Edsger
Write-Host "Building Edsger with optimizations..." -ForegroundColor Green

# Clean build artifacts
if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
if (Test-Path "*.egg-info") { Remove-Item -Recurse -Force "*.egg-info" }

# Check for GCC
$hasGCC = $false
try {
    gcc --version | Out-Null
    $hasGCC = $true
    Write-Host "Using GCC compiler" -ForegroundColor Green
} catch {
    Write-Host "Using MSVC compiler" -ForegroundColor Yellow
}

# Build with appropriate compiler
if ($hasGCC) {
    $env:CC = "gcc"
    $env:CFLAGS = "-Ofast -march=native"
} else {
    $env:CL = "/O2 /arch:AVX2"
}

# Build and install
python -m pip install -e . --no-build-isolation

Write-Host "Build complete!" -ForegroundColor Green