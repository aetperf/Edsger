# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../../src/"))

# -- Project information -----------------------------------------------------

project = "Edsger"
author = "François Pacull"
copyright = "2022-2026, Architecture & Performance"

try:
    from edsger._version import __version__

    release = __version__
except ImportError:
    release = "0.2.0"
# Short X.Y.Z version, without the local part of development versions
version = ".".join(release.split(".")[:3])

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_copybutton",
]

source_suffix = {".md": "markdown"}
exclude_patterns = []

myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

# -- API reference -----------------------------------------------------------

# Single backquotes in the docstrings are names of Python objects
default_role = "py:obj"
autodoc_member_order = "bysource"
autodoc_typehints = "none"
autodoc_class_signature = "mixed"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False
maximum_signature_line_length = 80

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "furo"
html_title = f"Edsger {version}"
html_show_sphinx = False
html_copy_source = False
html_show_sourcelink = False

html_theme_options = {
    "source_repository": "https://github.com/aetperf/Edsger",
    "source_branch": "release",
    "source_directory": "docs/source/",
    "navigation_with_keys": True,
    "light_css_variables": {
        "color-brand-primary": "#2f5d8a",
        "color-brand-content": "#2f5d8a",
    },
    "dark_css_variables": {
        "color-brand-primary": "#8fb8de",
        "color-brand-content": "#8fb8de",
    },
}

# Copy button: do not copy the prompts and the outputs of interactive sessions
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
