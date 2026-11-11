# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## Unreleased

### Performance Improvements
- **Dijkstra about 2x faster on road networks**: 2.45s to 1.24s on the USA DIMACS network (Linux, i9-12900H), both with and without decrease-key (`lazy=True`/`lazy=False`). Path lengths are bit-identical to the previous version.
- New inline 4-ary heap (`pq_4ary_inline.pxd`) used by `dijkstra.pyx` and `dijkstra_nodec.pyx`: keys stored in the heap nodes, branch-free selection of the smallest child, sentinel keys, hole-based sifts, heap operations inlined into the main loop
- Prefetching of the adjacency data of the next vertices to be scanned (heap root and its children)
- No per-vertex state array anymore: the path length array is the output array, written in place
- `# cython:` compiler directives moved before the module docstring in all the Cython modules: they were ignored by Cython where they were, so bounds checking was still active. BFS and Bellman-Ford are about 18% faster (E and NY DIMACS networks)

### Bug Fixes
- Fixed a heap overflow (memory corruption) of the lazy Dijkstra on dense graphs, where the number of heap entries can exceed the vertex count: the heap now grows when needed
- Negative vertex indices without `permute=True` now raise a `ValueError` when the graph is built, instead of silently producing wrong results
- `compute_path` (and so `get_path`) now raises an `IndexError` for an out-of-bounds vertex index, instead of returning an empty path
- `get_path` on a graph built with `permute=True` (`Dijkstra`, `BellmanFord`, `BFS`): it now returns the right path after `run(..., return_series=True)`, supports negative vertex labels, uses a memory that does not depend on the largest label, and raises a `ValueError` for a vertex that is not in the graph

### Input Validation
- `Dijkstra` now always rejects negative and NaN edge weights with a `ValueError` when the graph is built, whatever the value of `check_edges`. The algorithm is not valid with negative weights; use `BellmanFord` instead
- The Cython functions that can be called from Python (`dijkstra`, `dijkstra_nodec`, `bellman_ford`, `bfs`, `spiess_florian`, `path_tracking`) now check their arguments explicitly, since bounds checking is disabled: `IndexError` for a vertex index out of bounds, `ValueError` for inconsistent array lengths
- The lazy Dijkstra routines stop with a `ValueError` when called directly on a graph with negative weights on which they would not terminate
- New `edsger.commons.validate_graph(indptr, indices, data=None, nonnegative=True)`: optional full check of a graph in CSR/CSC format (pointers, indices, weights), in linear time (35 ms on the USA network). The algorithms do not call it: it is meant to be run once on arrays that do not come from `edsger.path`, before any number of runs

### Internal
- `dijkstra.pyx` and `dijkstra_nodec.pyx` now share their code: both include `dijkstra_core.pxi` (algorithm, public functions, embedded tests), compiled with a constant `LAZY` set to 0 or 1. The heap has a single pair of sift functions, with or without decrease-key. No change in speed or results
- Removed the `pq_4ary_nodec_0b` module, which was not used anymore
- cython-lint also runs on `.pxi` files (pre-commit, CI)

## Version 0.1.7 (2026-01-05)

### Performance Improvements
- **Major Windows optimization**: Dijkstra performance on Windows improved by 2.2x (6.0s to 2.75s on USA DIMACS network)
- **Windows/Linux gap reduced**: From 2.4x to just 1.07x (7% difference)
- Added memory prefetching to heap operations (`_min_heapify`, `_decrease_key_from_node_index`)
- Extended `prefetch_compat.h` with branch prediction macros and additional prefetch hints (T1, T2, NTA)
- Added `/Qspectre-` flag to MSVC compiler for benchmark builds

### Python Version Support
- **Added**: Python 3.14 support
- **Removed**: Python 3.9 support
- Supported versions: 3.10, 3.11, 3.12, 3.13, 3.14

### Bug Fixes
- Fixed graph-tool API compatibility (explicit `topology` submodule import)

### Type Checking
- Added `.pyi` stub files for all Cython modules to support static type checkers (pyright, ty)

### Build & CI
- Updated cibuildwheel to >=3.0.0
- Updated GitHub Actions to setup-python@v5
- Pinned benchmark script dependencies

## Version 0.1.6 (2025-10-12)

### New Features
- Added Breadth-First Search (BFS) algorithm
- BFS class with configurable sentinel value for unreachable vertices
- Full test coverage against SciPy implementation

## Version 0.1.5 (2025-08-22)

### New Features
- Added Bellman-Ford algorithm with negative cycle detection
- Support for both CSR and CSC graph formats

### Technical Improvements
- Parallel edge handling with automatic minimum weight selection

## Version 0.1.4 (2025-08-05)

### Build & CI
- Windows Server 2019 runner retired, updated CI configuration
- Test coverage and documentation action setup
- Actions now run on release creation

## Version 0.1.3 (2025-06-27)

### New Features
- Early termination support for Dijkstra (stop when target nodes are reached)
- Returns path length only for termination nodes in early termination case

### Performance Improvements
- Added prefetching for single-source shortest path

### Documentation
- Added early termination section to documentation
- Improved small example in README

## Version 0.1.2 (2025-06-22)

### Performance Improvements
- Cross-platform prefetching compatibility header using SSE intrinsics on x86/x64, ARM intrinsics on ARM64
- Applied prefetching to Dijkstra algorithm

### Build & CI
- Fixed PyPI wheel publishing artifact naming
- Added Codecov coverage reporting and badge

## Version 0.1.1 (2025-06-22)

### Performance Improvements
- Added memory prefetching headers and functions
- Prefetch hints in edge iteration loops for `compute_sssp` and `compute_sssp_w_path`
- Added LTO flags (Link Time Optimization)
- Added `noexcept` declarations to priority queue function signatures

### Build & CI
- Added support for Python 3.9-3.13
- Skip i686 architecture builds
- Fixed cython-lint command for Windows compatibility

### Code Quality
- Added black code formatter
- Added cython-lint for Cython code quality enforcement
- Fixed all cython-lint violations

## Version 0.1.0 (2025-06-18)

- Major version bump
- Documentation moved to ReadTheDocs
- Updated GitHub Actions

## Version 0.0.x (2022-11-26 to 2025-02-03)

Initial development releases including:

- Core Dijkstra's algorithm implementation in Cython
- CSR (Compressed Sparse Row) graph representation
- Path tracking support
- Basic test suite
- PyPI package setup
