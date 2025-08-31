#!/bin/bash

# Clean build script for Edsger Cython project
# Use this when making significant changes to ensure clean rebuild

echo "Cleaning build artifacts..."

# Remove compiled Cython C files
find src/ -name "*.c" -delete
echo "✓ Removed generated C files"

# Remove compiled shared libraries
find src/ -name "*.so" -delete
echo "✓ Removed shared libraries"

# Remove build directory
rm -rf build/
echo "✓ Removed build directory"

# Remove Python cache
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
echo "✓ Removed Python cache"

# Remove Cython HTML annotation files (if any)
find . -name "*.html" -path "*/build/*" -delete 2>/dev/null

echo "Clean complete! Ready for fresh build."
echo "Run: python setup.py build_ext --inplace"