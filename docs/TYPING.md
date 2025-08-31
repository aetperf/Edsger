# Type Checking with Pyright

This project uses [Pyright](https://github.com/microsoft/pyright) for static type checking.

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements-dev.txt
   ```

2. **Run type checking**:
   ```bash
   # Basic mode (default)
   make typecheck
   # or directly
   pyright
   
   # Strict mode
   make typecheck-strict
   ```

## Configuration

The type checking configuration is defined in `pyrightconfig.json`. The project currently uses **basic** type checking mode to allow for gradual type hint adoption.

## Current Status

The project is in the process of adding type hints. Currently:

- **Type checking mode**: Basic
- **Coverage**: Minimal type hints in Python files
- **Cython files**: Excluded from type checking (`.pyx` and `.pxd` files)

## Known Issues

### Import Resolution
Some optional dependencies may show import warnings:
- `scipy` - Used in tests and scripts
- `loguru` - Used in scripts
- `graph_tool` - Optional dependency for benchmarks
- `networkit` - Optional dependency for benchmarks

These can be safely ignored if you're not using these features.

### Cython Integration
Cython-generated modules may show as missing imports. These are expected:
- Imports from `.pyx` files will show warnings
- This is normal as Pyright cannot analyze Cython files directly

### Test Files
Some test files import functions directly from Cython modules that are not visible to Pyright.

## Adding Type Hints

When adding type hints to the project:

1. **Start with function signatures**:
   ```python
   def compute_path(edges: pd.DataFrame, source: int) -> np.ndarray:
       ...
   ```

2. **Use type aliases for clarity**:
   ```python
   from typing import TypeAlias
   
   NodeID: TypeAlias = int
   Weight: TypeAlias = float
   ```

3. **For Cython imports, use type ignore comments**:
   ```python
   from edsger.dijkstra import compute_sssp  # type: ignore
   ```

## VS Code Integration

If using VS Code, the project includes `.vscode/settings.json` with appropriate configurations for Pylance/Pyright integration.

## Pre-commit Hook

Pyright is configured as a pre-commit hook. It will run automatically before commits if you have pre-commit installed:

```bash
pre-commit install
```

To run manually:
```bash
pre-commit run pyright --all-files
```

## Gradual Typing Strategy

The project follows a gradual typing approach:

1. **Phase 1** (Current): Basic type checking, focus on obvious type errors
2. **Phase 2**: Add type hints to public APIs
3. **Phase 3**: Increase to standard mode, add more comprehensive type hints
4. **Phase 4**: Consider strict mode for new code

## Resources

- [Pyright Documentation](https://github.com/microsoft/pyright/blob/main/docs/configuration.md)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)
- [PEP 484 - Type Hints](https://www.python.org/dev/peps/pep-0484/)