# Edsger

*Graph algorithms in Cython*

Edsger is a Python library of path algorithms for directed graphs, written in [Cython](https://cython.org/). A graph is given as a dataframe of edges, from [pandas](https://pandas.pydata.org/) or [Polars](https://pola.rs/), and the results come back as [NumPy](https://numpy.org/) arrays or pandas Series. The library is named after Edsger W. Dijkstra.

| Algorithm | Class | Problem |
|---|---|---|
| Dijkstra | `Dijkstra` | Shortest paths with non-negative edge weights |
| Bellman-Ford | `BellmanFord` | Shortest paths with negative edge weights, negative cycle detection |
| Breadth-first search | `BFS` | Paths with the fewest edges |
| Spiess-Florian | `HyperpathGenerating` | Optimal strategies in frequency-based transit networks |

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

The result is the length of the shortest path from vertex 0 to each vertex of the graph.

## Performance

Dijkstra's algorithm takes about one second on the USA road network of the [9th DIMACS implementation challenge](https://archive.dimacs.rutgers.edu/Workshops/Challenge9/), which has 23.9 million vertices and 57.7 million edges.

```{image} assets/dijkstra_benchmark_comparison.png
:alt: Run time of Dijkstra's algorithm on the USA road network, for Edsger, NetworKit, graph-tool and SciPy
:width: 700px
```

Single-source shortest paths from one vertex of the USA road network, best of 5 runs. The Linux and Windows timings come from two different machines, described in [Benchmarks](benchmarks.md).

```{toctree}
:hidden:

installation
quickstart
benchmarks
api
changelog
contributing
```
