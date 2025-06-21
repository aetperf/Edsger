# Installation Guide

This guide will help you install the `edsger` package on your system.

## Requirements

Before you begin, ensure you have the following prerequisites:

- Python 3.11 or higher
- pip (Python package installer) or uv (recommended)
- (Optional) git for installing from source

## Method 1: Using Python venv + pip

Create a virtual environment and install with pip:

```bash
# Create a virtual environment
python -m venv edsger-env

# Activate the virtual environment
source edsger-env/bin/activate  # On Windows: edsger-env\Scripts\activate

# Install edsger
pip install edsger
```

## Method 2: Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) provides faster installation and better dependency resolution:

```bash
# Install uv if you haven't already
pip install uv

# Create a virtual environment with uv
uv venv edsger-env

# Activate the virtual environment
source edsger-env/bin/activate  # On Windows: edsger-env\Scripts\activate

# Install edsger
uv pip install edsger
```

## Installing from Source

If you prefer to install from source or contribute to development:

### 1. Clone the Repository

```bash
git clone https://github.com/aetperf/Edsger.git
cd Edsger
```

### 2. Using Python venv + pip

```bash
# Create and activate virtual environment
python -m venv edsger-env
source edsger-env/bin/activate  # On Windows: edsger-env\Scripts\activate

# Install in editable mode
pip install -e .
```

### 3. Using uv (Recommended)

```bash
# Create and activate virtual environment
uv venv edsger-env
source edsger-env/bin/activate  # On Windows: edsger-env\Scripts\activate

# Install in editable mode
uv pip install -e .
```

### 4. Verify the Installation

Check that the installation was successful by importing the package in a Python shell:

```python
python
>>> import edsger
>>> edsger.__version__
```

You should see the version number of the `edsger` package.

## Development Installation

For contributors who need development dependencies:

### Using Python venv + pip

```bash
# Create and activate virtual environment
python -m venv edsger-env
source edsger-env/bin/activate  # On Windows: edsger-env\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt
pip install -e .
```

### Using uv (Recommended)

```bash
# Create and activate virtual environment
uv venv edsger-env
source edsger-env/bin/activate  # On Windows: edsger-env\Scripts\activate

# Install development dependencies
uv pip install -r requirements-dev.txt
uv pip install -e .
```

## Troubleshooting

### Module Not Found Error

If you encounter a `ModuleNotFoundError`, make sure that:
1. The `edsger` package is installed correctly
2. You're using the correct Python environment
3. The `PYTHONPATH` is set appropriately (if needed)

### Compilation Issues

If you experience compilation issues, ensure you have:
- A working C compiler
- NumPy installed
- Cython >= 3.0 installed

## Uninstallation

To uninstall the `edsger` package:

```bash
pip uninstall edsger
```

## Getting Help

For further assistance:
- Check the [documentation](index.md)
- Open an issue on [GitHub](https://github.com/aetperf/Edsger)
- Contact the maintainer at [francois.pacull@architecture-performance.fr](mailto:francois.pacull@architecture-performance.fr)