# Edsger

*Graph algorithms in Cython*

Edsger is a Python library of graph algorithms for directed graphs, written in [Cython](https://cython.org/). A graph is given as a dataframe of edges, from [pandas](https://pandas.pydata.org/) or [Polars](https://pola.rs/), and the results come back as [NumPy](https://numpy.org/) arrays, pandas Series or Polars DataFrames. The library is named after Edsger W. Dijkstra.

| Family | Classes |
|---|---|
| Shortest paths | `Dijkstra`, `BellmanFord`, `AStar`, `BidirectionalDijkstra` |
| Traversal and components | `BFS`, `DFS`, `ConnectedComponents` |
| Spanning trees and centrality | `MinimumSpanningTree`, `Centrality` (PageRank, HITS, Katz) |
| Transit assignment | `HyperpathGenerating` (Spiess and Florian's optimal strategies) |

## Example

```python
import pandas as pd

from edsger import Dijkstra

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
:caption: Getting started

installation
quickstart
coming_from
```

```{toctree}
:hidden:
:caption: User guide

guide/shortest_paths
guide/traversal
guide/network_analysis
guide/transit
```

```{toctree}
:hidden:
:caption: Examples

examples/road_routing
examples/od_matrix
examples/transit_assignment
examples/ranking
```

```{toctree}
:hidden:
:caption: Reference

api
benchmarks
changelog
contributing
```
