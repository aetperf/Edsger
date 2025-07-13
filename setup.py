"""setup python file for edsger."""

import platform
import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

# Platform-specific compiler optimizations
if platform.system() == "Windows":
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

    # Try to add AVX2 support if available (might not be supported on all Windows versions)
    try:
        import subprocess

        # Check if we can use AVX2
        extra_compile_args.append("/arch:AVX2")
    except:
        # Fall back to SSE2 which is widely supported
        extra_compile_args.append("/arch:SSE2")

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

print(f"Building for {platform.system()} with compiler args: {extra_compile_args}")
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
