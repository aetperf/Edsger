# Type checking with ty

This project uses [ty](https://github.com/astral-sh/ty) for static type checking.

## Setup

```bash
pip install -r requirements-dev.txt
ty check
```

`ty check` also runs as a pre-commit hook and in the test workflow.

## Configuration

The configuration is in the `[tool.ty]` section of `pyproject.toml`: the `src` directory is checked, for Python 3.10, the oldest supported version. The Cython modules are typed through their `.pyi` stubs, next to the `.pyx` files.
