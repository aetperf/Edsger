"""setup python file for edsger."""

import platform
import subprocess
import sys
import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup


def detect_gcc_on_windows():
    """Detect if GCC (MinGW) is available on Windows."""
    if platform.system() != "Windows":
        return False

    try:
        result = subprocess.run(
            ["gcc", "--version"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            gcc_version = result.stdout.split("\n")[0]
            print(f"[INFO] Detected GCC on Windows: {gcc_version}")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass

    print("[INFO] GCC not found on Windows, using MSVC")
    return False


# Platform-specific compiler optimizations
if platform.system() == "Windows":
    # Try to use GCC first (like SciPy does), fall back to MSVC
    if detect_gcc_on_windows():
        print("[INFO] Using GCC toolchain on Windows (same as SciPy)")
        # Use same GCC flags as Linux for consistency with SciPy
        extra_compile_args = [
            "-Ofast",
            "-flto",
            "-march=native",
            "-ffast-math",
            "-funroll-loops",
        ]
        extra_link_args = ["-flto"]
        compiler_type = "GCC"
    else:
        print("[INFO] Using MSVC toolchain on Windows")
        # MSVC compiler flags for Windows
        extra_compile_args = [
            "/O2",  # Maximum optimizations (speed)
            "/Ot",  # Favor fast code over small code
            "/GL",  # Whole program optimization
            "/favor:INTEL64",  # Optimize for 64-bit Intel processors
            "/fp:fast",  # Fast floating-point model
            "/GS-",  # Disable buffer security checks for performance
            "/Gy",  # Enable function-level linking
        ]
        extra_link_args = [
            "/LTCG",  # Link-time code generation
            "/OPT:REF",  # Remove unreferenced functions/data
            "/OPT:ICF",  # Enable COMDAT folding
        ]

        # Try to add AVX2 support if available
        try:
            extra_compile_args.append("/arch:AVX2")
        except:
            # Fall back to SSE2 which is widely supported
            extra_compile_args.append("/arch:SSE2")

        compiler_type = "MSVC"

elif platform.system() == "Darwin":
    # Clang flags for macOS
    extra_compile_args = [
        "-Ofast",
        "-flto",
        "-march=native",
        "-ffast-math",
        "-funroll-loops",
    ]
    extra_link_args = ["-flto"]
    compiler_type = "Clang"
else:
    # GCC flags for Linux and other Unix-like systems
    extra_compile_args = [
        "-Ofast",
        "-flto",
        "-march=native",
        "-ffast-math",
        "-funroll-loops",
    ]
    extra_link_args = ["-flto"]
    compiler_type = "GCC"

print(f"Building for {platform.system()} with {compiler_type}")
print(f"Compiler args: {extra_compile_args}")
print(f"Link args: {extra_link_args}")

extensions = [
    Extension(
        "edsger.commons",
        ["src/edsger/commons.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "edsger.pq_4ary_dec_0b",
        ["src/edsger/pq_4ary_dec_0b.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "edsger.dijkstra",
        ["src/edsger/dijkstra.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "edsger.spiess_florian",
        ["src/edsger/spiess_florian.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "edsger.star",
        ["src/edsger/star.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "edsger.path_tracking",
        ["src/edsger/path_tracking.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
]

with open("requirements.txt") as fp:
    install_requires = fp.read().strip().split("\n")


def setup_package():
    setup(
        ext_modules=cythonize(
            extensions,
            compiler_directives={
                "language_level": "3",
                "boundscheck": False,
                "wraparound": False,
                "embedsignature": False,
                "cdivision": True,
                "initializedcheck": False,
            },
            include_path=["src/edsger/"],
        ),
        install_requires=install_requires,
        include_dirs=[np.get_include()],
    )


if __name__ == "__main__":
    setup_package()
