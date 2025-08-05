"""setup python file for edsger."""

import os

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

extra_compile_args = ["-Ofast", "-flto"]
extra_link_args = ["-flto"]

# Add coverage flags when building with CYTHON_TRACE
if os.environ.get("CYTHON_TRACE", "0") == "1":
    extra_compile_args = ["-O0", "-g"]  # Disable optimization for accurate coverage
    extra_link_args = []

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
    # Enable coverage for Cython when CYTHON_TRACE is set
    compiler_directives = {
        "language_level": "3",
        "boundscheck": False,
        "wraparound": False,
        "embedsignature": False,
        "cdivision": True,
        "initializedcheck": False,
    }

    if os.environ.get("CYTHON_TRACE", "0") == "1":
        compiler_directives["linetrace"] = True
        compiler_directives["profile"] = True
        print("Building with coverage support for Cython code")

    setup(
        ext_modules=cythonize(
            extensions,
            compiler_directives=compiler_directives,
            include_path=["src/edsger/"],
        ),
        install_requires=install_requires,
        include_dirs=[np.get_include()],
    )


if __name__ == "__main__":
    setup_package()
