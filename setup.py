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
                "-O3",  # Maximum optimization (or use -Ofast for even more aggressive)
                "-march=native",  # Optimize for the current CPU architecture
                "-mtune=native",  # Tune for the current CPU
                "-flto",  # Link-time optimization
                "-ffast-math",  # Fast floating-point math
                "-funroll-loops",  # Unroll loops for better performance
                "-ftree-vectorize",  # Enable auto-vectorization
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

            print(f"macOS build detection:")
            print(f"  sysconfig.get_platform(): {platform_str}")
            print(f"  ARCHFLAGS: {archflags if archflags else '(not set)'}")
            print(
                f"  _PYTHON_HOST_PLATFORM: {host_platform if host_platform else '(not set)'}"
            )

            # Check sysconfig.get_platform() (most reliable for pip builds)
            if "universal2" in platform_str:
                is_universal_build = True
                print(f"  -> Universal binary detected via sysconfig.get_platform()")
            # Check ARCHFLAGS environment variable (used by pip/setuptools)
            elif archflags.count("-arch") > 1 or "universal2" in archflags:
                is_universal_build = True
                print(f"  -> Universal binary detected via ARCHFLAGS")
            # Also check _PYTHON_HOST_PLATFORM for cibuildwheel
            elif "universal2" in host_platform:
                is_universal_build = True
                print(f"  -> Universal binary detected via _PYTHON_HOST_PLATFORM")
            else:
                print(f"  -> Native build (not universal binary)")

        if is_universal_build:
            print(
                f"Building with {compiler_name} optimizations on {system} (universal binary)"
            )
            # Skip -march=native for universal binaries (incompatible with cross-compilation)
            compile_args = ["-Ofast", "-flto"]
        else:
            print(f"Building with {compiler_name} optimizations on {system}")
            compile_args = ["-Ofast", "-flto", "-march=native"]

        link_args = ["-flto"]
        return compile_args, link_args, []
    # Unknown platform, use conservative flags
    print(f"Building on unknown platform {system}, using conservative optimizations")
    compile_args = ["-O2"]
    link_args = []
    return compile_args, link_args, []


# Get platform-specific compiler flags
extra_compile_args, extra_link_args, platform_macros = get_compiler_flags()

# Combine numpy macros with platform-specific macros
define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")] + platform_macros

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
        define_macros=define_macros,
    ),
    Extension(
        "edsger.dijkstra",
        ["src/edsger/dijkstra.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=define_macros,
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
]

with open("requirements.txt", encoding="utf-8") as fp:
    install_requires = fp.read().strip().split("\n")


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
        install_requires=install_requires,
        include_dirs=[np.get_include()],
    )


if __name__ == "__main__":
    setup_package()
