"""
    Setup file for edsger.
"""

import os
import re
from setuptools import setup

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

requirements = ["cython", "numpy"]
test_requirements = ["pytest"]

here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    with open(os.path.join(here, *parts), "r") as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


# Get the licence
with open("LICENSE") as f:
    license = f.read()

extra_compile_args = ["-Ofast"]

extensions = [
    Extension(
        "edsger.pqueue_imp_bin_dec_1b",
        ["src/edsger/pqueue_imp_bin_dec_1b.pyx"],
        extra_compile_args=extra_compile_args,
    ),
]

setup(
    name="edsger",
    version=find_version("src", "edsger", "__init__.py"),
    description="Graph-related algorithms in Python ",
    author="Francois Pacull",
    author_email="pacullfrancois@gmail.com",
    license=license,
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    package_data={
        "priority_queues.pqueue_imp_bin_dec_1b": ["src/edsger/pqueue_imp_bin_dec_1b.pxd"],
    },
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
        include_path=["src/edsger/"],
    ),
    install_requires=requirements,
    # setup_requires=setup_requirements,
    tests_require=test_requirements,
    extras_require={"test": test_requirements},
    include_dirs=[np.get_include()],
)

if __name__ == "__main__":
    try:
        setup(
            use_scm_version={
                "version_scheme": "no-guess-dev",
                "write_to": "_version.py",
                "write_to_template": '__version__ = "{version}"',
            }
        )
    except:  # noqa
        print(
            "\n\nAn error occurred while building the project, "
            "please ensure you have the most updated version of setuptools, "
            "setuptools_scm and wheel with:\n"
            "   pip install -U setuptools setuptools_scm wheel\n\n"
        )
        raise