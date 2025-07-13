#!/bin/bash
# Linux clean build script for Edsger
# This script performs a complete clean rebuild with optimizations

echo "==================================================================="
echo "EDSGER LINUX CLEAN BUILD WITH OPTIMIZATIONS"
echo "==================================================================="

# Function to print colored output
print_status() {
    echo -e "\033[1;32m[INFO]\033[0m $1"
}

print_warning() {
    echo -e "\033[1;33m[WARNING]\033[0m $1"
}

print_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

# Check if we're in the right directory
if [ ! -f "setup.py" ]; then
    print_error "setup.py not found. Please run this script from the Edsger root directory."
    exit 1
fi

print_status "Detected OS: $(uname -s)"
print_status "Python version: $(python --version)"

# Step 1: Clean all build artifacts
print_status "Cleaning all build artifacts..."

# Remove build directories
if [ -d "build" ]; then
    rm -rf build
    print_status "Removed build/ directory"
fi

if [ -d "dist" ]; then
    rm -rf dist
    print_status "Removed dist/ directory"
fi

# Remove egg-info directories
find . -name "*.egg-info" -type d -exec rm -rf {} + 2>/dev/null
print_status "Removed *.egg-info directories"

# Remove compiled Python files
find . -name "*.pyc" -delete 2>/dev/null
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
print_status "Removed *.pyc files and __pycache__ directories"

# Remove Cython-generated C files and compiled extensions
find src/edsger/ -name "*.c" -delete 2>/dev/null
find src/edsger/ -name "*.so" -delete 2>/dev/null
find src/edsger/ -name "*.html" -delete 2>/dev/null
print_status "Removed Cython-generated C files and compiled extensions"

# Step 2: Set optimal environment variables for GCC
print_status "Setting GCC optimization environment variables..."

export CFLAGS="-Ofast -flto -march=native -ffast-math -funroll-loops"
export CXXFLAGS="-Ofast -flto -march=native -ffast-math -funroll-loops" 
export LDFLAGS="-flto"

print_status "CFLAGS: $CFLAGS"
print_status "CXXFLAGS: $CXXFLAGS"
print_status "LDFLAGS: $LDFLAGS"

# Step 3: Build extensions in place
print_status "Building Cython extensions in place..."
python setup.py build_ext --inplace --force

if [ $? -ne 0 ]; then
    print_error "Build failed! Check the error messages above."
    exit 1
fi

# Step 4: Install in development mode
print_status "Installing package in development mode..."
pip install -e . --force-reinstall --no-deps

if [ $? -ne 0 ]; then
    print_error "Installation failed! Check the error messages above."
    exit 1
fi

# Step 5: Verify installation
print_status "Verifying installation..."
python -c "import edsger; print(f'Edsger version: {edsger.__version__}')" 2>/dev/null

if [ $? -eq 0 ]; then
    print_status "✓ Edsger successfully installed and importable"
else
    print_warning "⚠ Edsger installed but import test failed"
fi

echo ""
echo "==================================================================="
echo "CLEAN BUILD COMPLETED SUCCESSFULLY"
echo "==================================================================="
echo "Next steps:"
echo "  cd scripts"
echo "  python benchmark_comparison_os.py -r 5"
echo ""
echo "Expected Linux performance: ~2.5 seconds"
echo "==================================================================="