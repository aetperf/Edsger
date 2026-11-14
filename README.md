
![Tests Status](https://github.com/aetperf/edsger/actions/workflows/tests.yml/badge.svg?event=release)
[![codecov](https://codecov.io/gh/aetperf/edsger/branch/release/graph/badge.svg)](https://codecov.io/gh/aetperf/edsger)
[![Documentation Status](https://readthedocs.org/projects/edsger/badge/?version=latest)](https://edsger.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://img.shields.io/pypi/v/edsger.svg?refresh=1)](https://pypi.org/project/edsger/)
[![Downloads](https://static.pepy.tech/badge/edsger)](https://pepy.tech/project/edsger)
[![Python 3.10 | 3.11 | 3.12 | 3.13 | 3.14](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](https://pypi.org/project/edsger/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Cython lint: cython-lint](https://img.shields.io/badge/cython--lint-enabled-brightgreen.svg)](https://github.com/MarcoGorelli/cython-lint)
[![Checked with pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Edsger

*Graph algorithms in Cython*

Edsger is a Python library of path algorithms for directed graphs, written in [Cython](https://cython.org/). A graph is given as a dataframe of edges, from [pandas](https://pandas.pydata.org/) or [Polars](https://pola.rs/), and the results come back as [NumPy](https://numpy.org/) arrays, pandas Series or Polars DataFrames.

| Algorithm | Class | Problem |
|---|---|---|
| Dijkstra | `Dijkstra` | Shortest paths with non-negative edge weights |
| Bellman-Ford | `BellmanFord` | Shortest paths with negative edge weights, negative cycle detection |
| Breadth-first search | `BFS` | Paths with the fewest edges |
| Spiess-Florian | `HyperpathGenerating` | Optimal strategies in frequency-based transit networks |

Documentation: [https://edsger.readthedocs.io/en/latest/](https://edsger.readthedocs.io/en/latest/)

## Installation

```bash
pip install edsger
```

Edsger requires Python 3.10 or later. Wheels are published for Linux (x86_64), Windows and macOS (arm64).

## Example

```python
import pandas as pd

from edsger.path import Dijkstra

edges = pd.DataFrame(
    {
        "tail": [0, 0, 1, 2, 2, 3],
        "head": [1, 2, 2, 3, 4, 4],
        "weight": [1.0, 4.0, 2.0, 1.5, 3.0, 1.0],
    }
)

dijkstra = Dijkstra(edges)
dijkstra.run(vertex_idx=0)
```

    array([0. , 1. , 3. , 4.5, 5.5])

The result is the length of the shortest path from vertex 0 to each vertex of the graph. The graph is built once, and `run` can be called any number of times with different vertices. With `path_tracking=True`, the paths themselves can be retrieved.

```python
dijkstra.run(vertex_idx=0, path_tracking=True)
dijkstra.get_path(vertex_idx=4)
```

    array([4, 3, 2, 1, 0], dtype=uint32)

The other classes are described in the [quick start](https://edsger.readthedocs.io/en/latest/quickstart.html).

## Performance

Dijkstra's algorithm takes about one second on the USA road network of the [9th DIMACS implementation challenge](https://archive.dimacs.rutgers.edu/Workshops/Challenge9/), which has 23.9 million vertices and 57.7 million edges.

<img src="https://raw.githubusercontent.com/aetperf/edsger/release/docs/source/assets/dijkstra_benchmark_comparison.png" alt="Run time of Dijkstra's algorithm on the USA road network, for Edsger, NetworKit, graph-tool and SciPy" width="700">

Single-source shortest paths from one vertex of the USA road network, best of 5 runs. The Linux and Windows timings come from two different machines. The method and the versions are given in the [benchmarks](https://edsger.readthedocs.io/en/latest/benchmarks.html) page.

## Contributing

Bug reports and pull requests are welcome on [GitHub](https://github.com/aetperf/Edsger/issues). See the [contributing](https://edsger.readthedocs.io/en/latest/contributing.html) page for the development setup.

## License

Edsger is distributed under the MIT License, see the [LICENSE](https://github.com/aetperf/Edsger/blob/release/LICENSE) file.

## Contact

François Pacull, [francois.pacull@architecture-performance.fr](mailto:francois.pacull@architecture-performance.fr)
