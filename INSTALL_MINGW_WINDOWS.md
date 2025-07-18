# Installing MinGW-w64 for Edsger on Windows

## Why MinGW-w64?
Using GCC (via MinGW-w64) instead of MSVC can improve Edsger's performance by 40-50% on Windows, bringing it closer to Linux performance levels.

## Installation Methods

### Method 1: Conda (Recommended)
```bash
# If you're using Anaconda/Miniconda
conda install -c conda-forge m2w64-toolchain

# Verify installation
gcc --version
g++ --version
```

### Method 2: MSYS2
1. Download MSYS2 from https://www.msys2.org/
2. Install and run MSYS2
3. In MSYS2 terminal:
```bash
pacman -S mingw-w64-x86_64-gcc
pacman -S mingw-w64-x86_64-gcc-fortran
```
4. Add to PATH: `C:\msys64\mingw64\bin`

### Method 3: Standalone MinGW-w64
1. Download from https://winlibs.com/
2. Extract to `C:\mingw64`
3. Add `C:\mingw64\bin` to system PATH

## Building Edsger with GCC
Once GCC is installed:
```bash
# Clean previous builds
pip uninstall edsger
rm -rf build/ dist/ *.egg-info

# Rebuild with GCC
pip install -e . --no-build-isolation

# Or for production install
pip install . --no-build-isolation
```

## Verify GCC is Being Used
During build, you should see:
```
[INFO] Detected GCC on Windows: gcc (GCC) x.x.x
[INFO] Using GCC toolchain on Windows (same as SciPy)
Building for Windows with GCC
```

## Troubleshooting

### "gcc not found" Error
- Ensure GCC is in your PATH
- Restart your terminal/IDE after installation
- Try `where gcc` to check if it's accessible

### Build Errors with GCC
- Make sure you have the 64-bit version of MinGW-w64
- If using conda, activate your environment first
- Try adding `--no-build-isolation` flag to pip

### Performance Not Improved
- Verify GCC was used during build (check build output)
- Run benchmarks multiple times to account for variance
- Ensure Windows Defender isn't scanning during benchmarks