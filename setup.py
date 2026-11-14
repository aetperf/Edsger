"""setup python file for edsger."""

import os
import platform
import sys
import sysconfig

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup


def is_mingw():
    """Check if we're using MinGW/GCC on Windows."""
    if platform.system() != "Windows":
        return False

    # Check if GCC is mentioned in sys.version (common for MinGW builds)
    if "GCC" in sys.version:
        return True

    # Check platform string for mingw
    platform_str = sysconfig.get_platform()
    if "mingw" in platform_str.lower():
        return True

    return False


def is_portable_build():
    """Check if the binaries are built to be distributed (cibuildwheel).

    Such binaries must run on any CPU of the target architecture, so they must
    not be tuned for the build machine with -march=native: the 0.1.7 Linux
    wheels were built on an AVX-512 machine and crashed at import time
    (illegal instruction) on the CPUs without AVX-512.
    """
    return os.environ.get("CIBUILDWHEEL", "0") == "1"


def get_compiler_flags():
    """Get platform and compiler specific optimization flags."""
    # Check if we're building with coverage support
    if os.environ.get("CYTHON_TRACE", "0") == "1":
        print("Building with coverage support (CYTHON_TRACE=1)")
        return ["-O0", "-g"], [], []  # Debug flags for accurate coverage

    system = platform.system()

    if system == "Windows":
        if is_mingw():
            # MinGW/GCC on Windows - Aggressive optimizations matching Linux performance
            print("Building with MinGW/GCC optimizations on Windows (aggressive)")
            compile_args = [
                "-Ofast",  # Maximum optimization with fast-math
                "-march=native",  # Optimize for the current CPU architecture
                "-mtune=native",  # Tune for the current CPU
                "-flto",  # Link-time optimization
                "-funroll-loops",  # Unroll loops for better performance
                "-ftree-vectorize",  # Enable auto-vectorization
                "-fno-semantic-interposition",  # Allow more aggressive inlining
                "-msse4.2",  # Enable SSE4.2 instructions (most modern CPUs support this)
                "-mavx2",  # Enable AVX2 if your CPU supports it (comment out if not)
            ]
            link_args = [
                "-flto",  # Link-time optimization
                "-Wl,-O1",  # Optimize linker output
            ]
            # Windows-specific macros for better performance
            windows_macros = [
                ("WIN32_LEAN_AND_MEAN", None),  # Exclude rarely-used APIs
                ("NOMINMAX", None),  # Prevent min/max macro conflicts
                ("_USE_MATH_DEFINES", None),  # Enable math constants
                ("NDEBUG", None),  # Disable debug assertions
            ]
            return compile_args, link_args, windows_macros
        else:
            # MSVC on Windows - Aggressive optimizations for maximum performance
            print("Building with MSVC optimizations on Windows (aggressive)")
            compile_args = [
                "/O2",  # Maximum speed optimization
                "/Ob2",  # Inline function expansion
                "/Oi",  # Enable intrinsic functions
                "/Ot",  # Favor fast code
                "/Oy",  # Omit frame pointers
                "/GL",  # Whole program optimization
                "/fp:fast",  # Fast floating-point model
                "/arch:AVX2",  # Use AVX2 instructions (or /arch:SSE2 for older CPUs)
                "/favor:INTEL64",  # Optimize for Intel 64-bit processors
                "/GS-",  # Disable security checks for performance
                "/Gw",  # Optimize global data
                "/Qspectre-",  # Disable Spectre mitigations for performance
            ]
            link_args = [
                "/LTCG",  # Link-time code generation
                "/OPT:REF",  # Eliminate unreferenced functions/data
                "/OPT:ICF",  # Enable COMDAT folding
            ]
            # Windows-specific macros for better performance
            windows_macros = [
                ("WIN32_LEAN_AND_MEAN", None),  # Exclude rarely-used APIs
                ("NOMINMAX", None),  # Prevent min/max macro conflicts
                ("_USE_MATH_DEFINES", None),  # Enable math constants
                ("_CRT_SECURE_NO_WARNINGS", None),  # Disable CRT warnings
                ("NDEBUG", None),  # Disable debug assertions
            ]
        return compile_args, link_args, windows_macros
    if system in ["Linux", "Darwin"]:
        # Linux or macOS with GCC/Clang
        compiler_name = "GCC/Clang" if system == "Linux" else "Clang"

        # Check if building a universal binary on macOS
        is_universal_build = False
        if system == "Darwin":
            # Debug output to diagnose universal binary detection
            platform_str = sysconfig.get_platform()
            archflags = os.environ.get("ARCHFLAGS", "")
            host_platform = os.environ.get("_PYTHON_HOST_PLATFORM", "")

            print("macOS build detection:")
            print(f"  sysconfig.get_platform(): {platform_str}")
            print(f"  ARCHFLAGS: {archflags if archflags else '(not set)'}")
            print(
                f"  _PYTHON_HOST_PLATFORM: {host_platform if host_platform else '(not set)'}"
            )

            # Check sysconfig.get_platform() (most reliable for pip builds)
            if "universal2" in platform_str:
                is_universal_build = True
                print("  -> Universal binary detected via sysconfig.get_platform()")
            # Check ARCHFLAGS environment variable (used by pip/setuptools)
            elif archflags.count("-arch") > 1 or "universal2" in archflags:
                is_universal_build = True
                print("  -> Universal binary detected via ARCHFLAGS")
            # Also check _PYTHON_HOST_PLATFORM for cibuildwheel
            elif "universal2" in host_platform:
                is_universal_build = True
                print("  -> Universal binary detected via _PYTHON_HOST_PLATFORM")
            else:
                print("  -> Native build (not universal binary)")

        if is_universal_build:
            print(
                f"Building with {compiler_name} optimizations on {system} (universal binary)"
            )
            # Skip -march=native for universal binaries (incompatible with cross-compilation)
            compile_args = ["-Ofast", "-flto"]
        else:
            compile_args = [
                "-Ofast",
                "-flto",
                "-fno-plt",  # Avoid PLT indirection for faster calls
                "-fno-semantic-interposition",  # Allow more aggressive inlining
                "-funroll-loops",  # More aggressive loop unrolling
            ]
            if is_portable_build():
                print(
                    f"Building with {compiler_name} optimizations on {system} "
                    "(portable binary, baseline instruction set)"
                )
            else:
                print(f"Building with {compiler_name} optimizations on {system}")
                # Tune for the CPU of the build machine: local builds only
                compile_args.append("-march=native")

        link_args = ["-flto"]
        return compile_args, link_args, []
    # Unknown platform, use conservative flags
    print(f"Building on unknown platform {system}, using conservative optimizations")
    compile_args = ["-O2"]
    link_args = []
    return compile_args, link_args, []


# Get platform-specific compiler flags
extra_compile_args, extra_link_args, platform_macros = get_compiler_flags()

# -Ofast implies -ffinite-math-only, with which the glibc headers of the
# manylinux images (2.28) redirect the math functions to their *_finite
# variants, e.g. asin to __asin_finite, which recent glibc versions do not
# export anymore: the A* module, which calls asin for the haversine distance,
# then fails to import. MSVC does not use these flags.
no_finite_math_only = (
    ["-fno-finite-math-only"]
    if any(flag in ("-Ofast", "-ffast-math") for flag in extra_compile_args)
    else []
)

# Combine numpy macros with platform-specific macros
# Large Pages / THP support is always enabled
define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")] + platform_macros
define_macros_with_thp = define_macros + [("USE_LARGE_PAGES", None)]
# The Windows large pages allocator enables SeLockMemoryPrivilege (Advapi32)
thp_libraries = ["advapi32"] if platform.system() == "Windows" else []

extensions = [
    Extension(
        "edsger.commons",
        ["src/edsger/commons.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=define_macros,
    ),
    Extension(
        "edsger.pq_4ary_dec_0b",
        ["src/edsger/pq_4ary_dec_0b.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=define_macros_with_thp,
        libraries=thp_libraries,
    ),
    Extension(
        "edsger.dijkstra",
        ["src/edsger/dijkstra.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=define_macros_with_thp,
        libraries=thp_libraries,
    ),
    Extension(
        "edsger.dijkstra_nodec",
        ["src/edsger/dijkstra_nodec.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=define_macros_with_thp,
        libraries=thp_libraries,
    ),
    Extension(
        "edsger.spiess_florian",
        ["src/edsger/spiess_florian.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=define_macros,
    ),
    Extension(
        "edsger.star",
        ["src/edsger/star.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=define_macros,
    ),
    Extension(
        "edsger.path_tracking",
        ["src/edsger/path_tracking.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=define_macros,
    ),
    Extension(
        "edsger.bellman_ford",
        ["src/edsger/bellman_ford.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=define_macros,
    ),
    Extension(
        "edsger.bfs",
        ["src/edsger/bfs.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=define_macros,
    ),
    Extension(
        "edsger.dfs",
        ["src/edsger/dfs.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=define_macros,
    ),
    Extension(
        "edsger.components_core",
        ["src/edsger/components_core.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=define_macros,
    ),
    Extension(
        "edsger.kruskal",
        ["src/edsger/kruskal.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=define_macros,
    ),
    Extension(
        "edsger.power_iteration",
        ["src/edsger/power_iteration.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=define_macros,
    ),
    Extension(
        "edsger.astar",
        ["src/edsger/astar.pyx"],
        extra_compile_args=extra_compile_args + no_finite_math_only,
        extra_link_args=extra_link_args,
        define_macros=define_macros_with_thp,
        libraries=thp_libraries,
    ),
    Extension(
        "edsger.bidirectional_dijkstra",
        ["src/edsger/bidirectional_dijkstra.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=define_macros_with_thp,
        libraries=thp_libraries,
    ),
]


def setup_package():
    """Configure and run the package setup with Cython extensions."""
    # Compiler directives are defined at the top of each .pyx file
    if os.environ.get("CYTHON_TRACE", "0") == "1":
        print("Building with CYTHON_TRACE=1 (coverage mode)")

    setup(
        ext_modules=cythonize(
            extensions,
            include_path=["src/edsger/"],
        ),
        include_dirs=[np.get_include()],
    )


if __name__ == "__main__":
    setup_package()
