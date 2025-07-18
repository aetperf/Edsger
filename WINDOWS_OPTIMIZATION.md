# Windows Performance Optimization Guide for Edsger

## Performance Gap Analysis
Current performance: **2.45x slower on Windows vs Linux** (6.032s vs 2.464s on USA road network)

## Optimization Strategies

### 1. Use GCC Compiler (Primary Solution)
The most significant improvement comes from using GCC instead of MSVC:

```bash
# Install MinGW-w64 via conda
conda install -c conda-forge m2w64-toolchain

# Verify installation
gcc --version

# Rebuild Edsger
pip install -e . --force-reinstall
```

Expected improvement: **~40-50%** (reducing gap to 1.4-1.6x)

### 2. Profile-Guided Optimization (PGO)
Add PGO support for both MSVC and GCC:

```python
# In setup.py, add for MSVC:
extra_compile_args.extend(["/GL", "/Qpar"])  # Whole program optimization + auto-parallelization
extra_link_args.extend(["/LTCG:PGO"])  # PGO support

# For GCC:
extra_compile_args.extend(["-fprofile-generate", "-fprofile-use"])
```

### 3. Windows Memory Allocation Optimization
Windows has different memory allocation patterns. Consider:

1. **Large Page Support**: Enable large pages for better TLB performance
2. **Memory Prefetching**: Already implemented in `prefetch_compat.h`
3. **NUMA Awareness**: For multi-socket systems

### 4. Thread Affinity and Priority
```python
# Add to your benchmark script
import os
if platform.system() == "Windows":
    # Set process to high priority
    import psutil
    p = psutil.Process(os.getpid())
    p.nice(psutil.HIGH_PRIORITY_CLASS)
    
    # Pin to specific CPU cores
    p.cpu_affinity([0, 1, 2, 3])  # Use first 4 cores
```

### 5. Disable Windows Defender Real-time Scanning
Windows Defender can impact performance by 10-20%. Temporarily exclude your working directory:
```powershell
Add-MpPreference -ExclusionPath "C:\Users\fpacu\Documents\Workspace\Edsger"
```

### 6. Build System Optimizations
Update `setup.py` for better Windows performance:

```python
if platform.system() == "Windows":
    if detect_gcc_on_windows():
        # GCC optimizations matching Linux
        extra_compile_args = [
            "-Ofast",
            "-flto",
            "-march=native",
            "-ffast-math",
            "-funroll-loops",
            "-fopenmp",  # OpenMP support
        ]
    else:
        # Enhanced MSVC optimizations
        extra_compile_args = [
            "/O2", "/Ot", "/GL", "/fp:fast",
            "/favor:INTEL64", "/Qpar",  # Auto-parallelization
            "/openmp",  # OpenMP support
            "/Qvec-report:2",  # Vectorization reporting
        ]
```

### 7. Cython Directive Optimizations
Add to your `.pyx` files:
```cython
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: overflowcheck=False
# cython: profile=False
```

## Expected Results
With all optimizations:
- **GCC**: 1.4-1.6x slower than Linux (vs current 2.45x)
- **MSVC + optimizations**: 1.8-2.0x slower than Linux

## Verification
After implementing optimizations, run:
```bash
cd scripts
python benchmark_comparison_os.py -n USA -r 5
```

Compare the new `benchmark_dimacs_USA_windows.json` with the previous results.