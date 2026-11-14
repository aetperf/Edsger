# Quick start

## A graph is a dataframe of edges

A graph is given as a dataframe with one row per directed edge: a `tail` (start) vertex, a `head` (end) vertex and a `weight`. Other column names can be passed with the `tail`, `head` and `weight` arguments. The dataframe can be pandas, with NumPy or Arrow data types, or Polars.

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
dijkstra
```

% output

    Dijkstra(n_vertices=5, n_edges=6, permute=False, orientation='out', lazy=True)

% /output

The graph is built once, when the object is created. The vertex identifiers are integers; see [Vertex identifiers](guide/shortest_paths.md#vertex-identifiers) when they are large, sparse or negative.

## Shortest path lengths

`run` returns the length of the shortest path from a vertex to every vertex, and can be called any number of times.

```python
dijkstra.run(vertex_idx=0)
```

% output

    array([0. , 1. , 3. , 4.5, 5.5])

% /output

Element `i` is the path length from vertex 0 to vertex `i`, `inf` if there is no path.

## Paths

With `path_tracking=True`, the shortest path tree is kept and `get_path` returns the vertices of a path, from the given vertex back to the source.

```python
dijkstra.run(vertex_idx=0, path_tracking=True)
dijkstra.get_path(vertex_idx=4)
```

% output

    array([4, 3, 2, 1, 0], dtype=uint32)

% /output

## pandas and Polars results

The `output` argument returns the results labelled by vertex, as a pandas Series or a Polars DataFrame.

```python
dijkstra.run(vertex_idx=0, output="polars")
```

% output

    shape: (5, 2)
    ┌────────────┬─────────────┐
    │ vertex_idx ┆ path_length │
    │ ---        ┆ ---         │
    │ i64        ┆ f64         │
    ╞════════════╪═════════════╡
    │ 0          ┆ 0.0         │
    │ 1          ┆ 1.0         │
    │ 2          ┆ 3.0         │
    │ 3          ┆ 4.5         │
    │ 4          ┆ 5.5         │
    └────────────┴─────────────┘

% /output

## Many sources at once

`run_many` runs from several vertices in parallel threads. With `termination_nodes`, the result is an origin-destination matrix.

```python
dijkstra.run_many([0, 1], termination_nodes=[3, 4], output="pandas")
```

% output

    destination    3    4
    origin
    0            4.5  5.5
    1            3.5  4.5

% /output

## Next steps

- The user guide has one page per family of algorithms: [shortest paths](guide/shortest_paths.md), [traversal and components](guide/traversal.md), [spanning trees and centrality](guide/network_analysis.md) and [transit assignment](guide/transit.md).
- The examples work through complete use cases: [routing on a road network](examples/road_routing.md), [an origin-destination matrix](examples/od_matrix.md), [a transit assignment](examples/transit_assignment.md) and [ranking the vertices of a network](examples/ranking.md).
- [Coming from NetworkX or SciPy](coming_from.md) maps their functions to Edsger.
