Installation Guide
==================

This guide will help you install the `edsger` package on your system.

Requirements
------------

Before you begin, ensure you have the following prerequisites:

- Python 3.6 or higher
- pip (Python package installer)

Installing from PyPI
--------------------

The easiest way to install `edsger` is by using `pip` to install it directly from PyPI. Open a terminal and run the following command::

   pip install edsger

This command will download and install the latest version of `edsger` along with its dependencies.

Installing from Source
----------------------

If you prefer to install `edsger` from source, follow these steps:

1. **Clone the Repository:**

   Clone the `edsger` repository from GitHub to your local machine::

      git clone https://github.com/aetperf/Edsger.git
      cd edsger

2. **Create a Virtual Environment:**

   It is recommended to use a virtual environment to manage dependencies. Create and activate a virtual environment using the following commands::

      python -m venv venv
      source venv/bin/activate   # On Windows use `venv\Scripts\activate`

3. **Install the Package:**

   You have two options to install the `edsger` package:

   - Install the package in editable mode to include all dependencies::

      pip install -e .

   - Or, install the dependencies from the `requirements.txt` file and then the package::

      pip install -r requirements.txt
      pip install .

4. **Verify the Installation:**

   Check that the installation was successful by importing the package in a Python shell::

      python
      >>> import edsger
      >>> edsger.__version__

   You should see the version number of the `edsger` package.

Troubleshooting
---------------

- **Module Not Found Error:**

  If you encounter a `ModuleNotFoundError`, make sure that the `edsger` package is installed correctly and that the `PYTHONPATH` is set appropriately::

     export PYTHONPATH=/path/to/your/edsger:$PYTHONPATH

- **Dependencies:**

  If there are issues with missing dependencies, ensure that all required packages are installed. You can list them in a `requirements.txt` file and install them using::

     pip install -r requirements.txt

Uninstallation
--------------

To uninstall the `edsger` package, simply run::

   pip uninstall edsger

Further Help
------------

For further assistance, refer to the documentation or contact the maintainer at `francois.pacull@architecture-performance.fr <mailto:francois.pacull@architecture-performance.fr>`_.
