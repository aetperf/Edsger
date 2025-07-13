# Clean Build Scripts for Optimal Performance

This directory contains clean build scripts that ensure Edsger is compiled with maximum performance optimizations for each platform.

## Why Use Clean Builds?

The performance gap between Windows and Linux can be significant due to:
- Incorrect compiler flags being used
- Build artifacts from previous compilations
- Incomplete rebuilds not applying new optimizations

**Performance Issues Observed:**
- Linux (GCC optimized): ~2.5 seconds
- Windows (unoptimized): ~6.0 seconds (**2.4x slower**)
- Windows (after partial optimization): ~5.0 seconds (**2.0x slower**)

## Clean Build Scripts

### Linux: `clean_build_linux.sh`

**Usage:**
```bash
./clean_build_linux.sh
```

**What it does:**
- Removes all build artifacts (build/, dist/, *.egg-info, *.pyc, *.so, *.c)
- Sets optimal GCC environment variables:
  - `CFLAGS="-Ofast -flto -march=native -ffast-math -funroll-loops"`
  - `CXXFLAGS="-Ofast -flto -march=native -ffast-math -funroll-loops"`
  - `LDFLAGS="-flto"`
- Forces complete rebuild with `--force` flag
- Installs in development mode with `--force-reinstall`
- Verifies installation

### Windows: `clean_build_windows.bat` / `clean_build_windows.ps1`

**Command Prompt:**
```cmd
clean_build_windows.bat
```

**PowerShell:**
```powershell
.\clean_build_windows.ps1
```

**What it does:**
- Removes all build artifacts (build/, dist/, *.egg-info, *.pyc, *.pyd, *.c)
- Sets optimal MSVC environment variables:
  - `CL=/O2 /Ot /GL /favor:INTEL64 /fp:fast /GS- /Gy /arch:AVX2`
  - `LINK=/LTCG /OPT:REF /OPT:ICF`
- Falls back to SSE2 if AVX2 fails
- Forces complete rebuild with `--force` flag
- Installs in development mode with `--force-reinstall`
- Verifies installation

## Compiler Flags Explained

### Linux/macOS (GCC/Clang)
- `-Ofast`: Maximum speed optimizations (including -O3)
- `-flto`: Link-time optimization
- `-march=native`: Optimize for current CPU architecture
- `-ffast-math`: Fast floating-point math (may reduce precision)
- `-funroll-loops`: Unroll loops for better performance

### Windows (MSVC)
- `/O2`: Maximum optimizations for speed
- `/Ot`: Favor speed over size
- `/GL`: Whole program optimization
- `/favor:INTEL64`: Optimize for 64-bit Intel processors
- `/fp:fast`: Fast floating-point model
- `/GS-`: Disable security checks for performance
- `/Gy`: Function-level linking
- `/arch:AVX2`: Use AVX2 SIMD instructions (falls back to SSE2)
- `/LTCG`: Link-time code generation
- `/OPT:REF`: Remove unreferenced code
- `/OPT:ICF`: Identical COMDAT folding

## Expected Performance Improvements

### Before Clean Build
- **Linux**: ~2.5s (already optimized)
- **Windows**: ~6.0s (unoptimized build)

### After Clean Build
- **Linux**: ~2.5s (should remain the same or slightly better)
- **Windows**: ~3.0-4.0s (**40-50% improvement expected**)

### Target Performance Gap
- **Current**: Windows 2.4x slower than Linux
- **Target**: Windows 1.2-1.5x slower than Linux

## Troubleshooting

### Windows Build Issues

**AVX2 Compilation Errors:**
```
Error: /arch:AVX2 not supported
```
**Solution:** The script automatically falls back to SSE2.

**Link-Time Optimization Errors:**
```
Error: LNK1000 or similar linker errors
```
**Solution:** Edit the script to remove `/GL` and `/LTCG` flags.

**General MSVC Issues:**
- Ensure Visual Studio Build Tools are installed
- Use "Developer Command Prompt" or "Developer PowerShell"

### Linux Build Issues

**GCC Not Found:**
```bash
sudo apt-get install build-essential
```

**Missing Dependencies:**
```bash
sudo apt-get install python3-dev
```

## Verification

After running the clean build, verify with benchmarks:

```bash
cd scripts
python benchmark_comparison_os.py -r 5
```

**Expected Results:**
- Linux: min_time ≈ 2.4-2.6 seconds
- Windows: min_time ≈ 3.0-4.0 seconds (down from 6.0s)

## Advanced Optimization

### Intel Compiler (If Available)
For maximum performance, consider Intel's compiler:

**Linux:**
```bash
export CC=icc
export CXX=icpc
./clean_build_linux.sh
```

**Windows:**
```cmd
set CC=icl
set CXX=icl
clean_build_windows.bat
```

### Profile-Guided Optimization (PGO)
For production builds, consider PGO:

1. Build with profiling: Add `/GL /LTCG:PGI` (Windows) or `-fprofile-generate` (Linux)
2. Run representative workload
3. Rebuild with profile data: `/LTCG:PGO` (Windows) or `-fprofile-use` (Linux)

## Files Overview

- `clean_build_linux.sh` - Linux/macOS bash script
- `clean_build_windows.bat` - Windows batch script  
- `clean_build_windows.ps1` - Windows PowerShell script
- `setup.py` - Modified with platform-specific optimizations
- `WINDOWS_OPTIMIZATION.md` - Detailed Windows optimization guide

## Performance Monitoring

Monitor the effectiveness of optimizations:

1. **Before/After Benchmarks**: Compare performance before and after clean build
2. **CPU Profiling**: Use tools like `perf` (Linux) or Visual Studio Profiler (Windows)
3. **SIMD Usage**: Verify that AVX2/SSE instructions are being used

The clean build process should significantly reduce the Windows performance gap while maintaining optimal Linux performance.