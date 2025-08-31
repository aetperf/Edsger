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
            # MinGW/GCC on Windows - Conservative optimizations for better performance
            print("Building with MinGW/GCC optimizations on Windows (conservative)")
            compile_args = [
                "-O2",
                "-march=x86-64",
                "-mtune=generic",
                "-msse2",
                "-ffast-math",
            ]
            link_args = []
            # Windows-specific macros for better performance
            windows_macros = [
                ("WIN32_LEAN_AND_MEAN", None),  # Exclude rarely-used APIs
                ("NOMINMAX", None),  # Prevent min/max macro conflicts
                ("_USE_MATH_DEFINES", None),  # Enable math constants
            ]
        else:
            # MSVC on Windows - Conservative optimizations for better performance
            print("Building with MSVC optimizations on Windows (conservative)")
            compile_args = ["/O2", "/fp:fast", "/arch:SSE2", "/favor:INTEL64"]
            link_args = []
            # Windows-specific macros for better performance
            windows_macros = [
                ("WIN32_LEAN_AND_MEAN", None),  # Exclude rarely-used APIs
                ("NOMINMAX", None),  # Prevent min/max macro conflicts
                ("_USE_MATH_DEFINES", None),  # Enable math constants
                (
                    "_CRT_SECURE_NO_WARNINGS",
                    None,
                ),  # Disable CRT warnings for performance
            ]
        return compile_args, link_args, windows_macros
    elif system in ["Linux", "Darwin"]:
        # Linux or macOS with GCC/Clang
        compiler_name = "GCC/Clang" if system == "Linux" else "Clang"
        print(f"Building with {compiler_name} optimizations on {system}")
        compile_args = ["-Ofast", "-flto", "-march=native"]
        link_args = ["-flto"]
        return compile_args, link_args, []
    else:
        # Unknown platform, use conservative flags
        print(
            f"Building on unknown platform {system}, using conservative optimizations"
        )
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
]

with open("requirements.txt") as fp:
    install_requires = fp.read().strip().split("\n")


def setup_package():
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
