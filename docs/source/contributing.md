# Contributing

Bug reports and pull requests are welcome on [GitHub](https://github.com/aetperf/Edsger/issues). A bug report is easier to handle with a small graph that reproduces the problem, the version of Edsger and the platform.

## Development setup

Building the package requires a C compiler.

```bash
git clone https://github.com/aetperf/Edsger.git
cd Edsger
pip install -r requirements-dev.txt
pip install -e .
```

The last command compiles the Cython modules. It has to be run again after a change in a `.pyx`, `.pxi` or `.pxd` file.

## Tests

```bash
pytest tests
```

The results of the shortest path algorithms are compared with those of SciPy, which is installed by `requirements-dev.txt`.

## Code style

Python code is linted and formatted with [ruff](https://github.com/astral-sh/ruff), Cython code is checked with [cython-lint](https://github.com/MarcoGorelli/cython-lint), and the type hints with [ty](https://github.com/astral-sh/ty). The three tools run as [pre-commit](https://pre-commit.com/) hooks.

```bash
pip install pre-commit
pre-commit install
```

## Documentation

The documentation is built with [Sphinx](https://www.sphinx-doc.org/) from the Markdown files of `docs/source` and from the docstrings.

```bash
pip install -r docs/requirements.txt
cd docs
make html
```

The pages are written to `docs/build/html`. User-visible changes are listed in `CHANGELOG.md`.
