
![Tests Status](https://github.com/aetperf/edsger/actions/workflows/tests.yml/badge.svg?branch=release)
[![codecov](https://codecov.io/gh/aetperf/edsger/branch/release/graph/badge.svg)](https://codecov.io/gh/aetperf/edsger)
[![PyPI version](https://img.shields.io/pypi/v/edsger.svg)](https://pypi.org/project/edsger/)
[![Downloads](https://static.pepy.tech/badge/edsger)](https://pepy.tech/project/edsger)
[![Python 3.9 | 3.10 | 3.11 | 3.12 | 3.13](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://pypi.org/project/edsger/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Cython lint: cython-lint](https://img.shields.io/badge/cython--lint-enabled-brightgreen.svg)](https://github.com/MarcoGorelli/cython-lint)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Edsger

*Graph algorithms in Cython*

Welcome to our Python library for graph algorithms. So far, the library only includes Dijkstra's algorithm but we should add a range of common path algorithms later. It is also open-source and easy to integrate with other Python libraries. To get started, simply install the library using pip, and import it into your Python project.

Documentation : [https://edsger.readthedocs.io/en/latest/](https://edsger.readthedocs.io/en/latest/)

## Small example : Dijkstra's Algorithm

To use Dijkstra's algorithm, you can import the `Dijkstra` class from the `path` module. The function takes a graph and a source node as input, and returns the shortest path from the source node to all other nodes in the graph.

```python
import pandas as pd

from edsger.path import Dijkstra

# Create a DataFrame with the edges of the graph
edges = pd.DataFrame({
    'tail': [0, 0, 1, 2, 2, 3],
    'head': [1, 2, 2, 3, 4, 4],
    'weight': [1, 4, 2, 1.5, 3, 1]
})
edges
```

    |    |   tail |   head |   weight |
    |---:|-------:|-------:|---------:|
    |  0 |      0 |      1 |      1.0 |
    |  1 |      0 |      2 |      4.0 |
    |  2 |      1 |      2 |      2.0 |
    |  3 |      2 |      3 |      1.5 |
    |  4 |      2 |      4 |      3.0 |
    |  5 |      3 |      4 |      1.0 |

```python
# Initialize the Dijkstra object
dijkstra = Dijkstra(edges)

# Run the algorithm from a source vertex
shortest_paths = dijkstra.run(vertex_idx=0)
print("Shortest paths:", shortest_paths)
```

    Shortest paths: [0.  1.  3.  4.5 5.5]

We get the shortest paths from the source node 0 to all other nodes in the graph. The output is an array with the shortest path length to each node. A path length is the sum of the weights of the edges in the path.

## Why Use Edsger?

Edsger is designed to be **dataframe-friendly**, providing seamless integration with pandas workflows for graph algorithms. Also it is rather efficient. Our benchmarks on the USA road network (23.9M vertices, 57.7M edges) demonstrate nice performance:

<img src="docs/source/assets/dijkstra_benchmark_comparison.png" alt="Dijkstra Performance Comparison" width="700">

*Benchmark performed on Intel i9-12900H laptop.*

## Contributing

We welcome contributions to the Edsger library. If you have any suggestions, bug reports, or feature requests, please open an issue on our [GitHub repository](https://github.com/aetperf/Edsger).

## License

Edsger is licensed under the MIT License. See the LICENSE file for more details.

## Contact

For any questions or inquiries, please contact François Pacull at [francois.pacull@architecture-performance.fr](mailto:francois.pacull@architecture-performance.fr).
