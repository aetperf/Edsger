"""setup python file for edsger.
"""

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

extra_compile_args = ["-Ofast"]

extensions = [
    Extension(
        "edsger.commons",
        ["src/edsger/commons.pyx"],
        extra_compile_args=extra_compile_args,
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "edsger.pqueue_imp_bin_dec_0b",
        ["src/edsger/pqueue_imp_bin_dec_0b.pyx"],
        extra_compile_args=extra_compile_args,
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
        include_path=["src/edsger/"],
    ),
    include_dirs=[np.get_include()],
)
