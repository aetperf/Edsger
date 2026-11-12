# Installation

Edsger requires Python 3.10 or later.

```bash
pip install edsger
```

NumPy and pandas are installed as dependencies. Two other packages are optional: [PyArrow](https://arrow.apache.org/docs/python/) for pandas dataframes with Arrow data types, and Polars for Polars dataframes.

```bash
pip install pyarrow polars
```

## Binary wheels

Wheels are published on [PyPI](https://pypi.org/project/edsger/) for Python 3.10 to 3.14 on the following platforms.

| System | Architecture | Compiler |
|---|---|---|
| Linux (glibc and musl) | x86_64 | Clang |
| Windows | x86_64, x86 | MSVC |
| macOS | arm64 | Apple Clang |

On the other platforms, pip builds the package from the source distribution, which requires a C compiler.

The Linux and macOS wheels only use the baseline instruction set of the architecture. The Windows wheels are compiled with `/arch:AVX2` and require a CPU with AVX2.

## From source

```bash
git clone https://github.com/aetperf/Edsger.git
cd Edsger
pip install .
```

A build from source on Linux or macOS uses `-march=native`: the binary is tuned for the CPU of the build machine and may not run on another one. The compiler is the default one of the Python installation. Another compiler can be selected with the `CC` environment variable.

```bash
CC=clang pip install .
```

On Linux, the Dijkstra code is about 10% faster when compiled with Clang than with GCC, see [Benchmarks](benchmarks.md).

## Check

```python
import edsger

edsger.__version__
```
