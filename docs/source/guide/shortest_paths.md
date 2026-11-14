# Shortest paths

| Class | Problem | Method |
|---|---|---|
| `Dijkstra` | From one vertex to all the others, or from all to one, non-negative weights | `run`, `run_many` |
| `BellmanFord` | Same, with negative weights; detects negative cycles | `run` |
| `AStar` | Between two vertices, guided by a heuristic | `run(source, target)` |
| `BidirectionalDijkstra` | Between two vertices | `run(source, target)` |

All the examples of this page use the following graph.

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

% output

    array([0. , 1. , 3. , 4.5, 5.5])

% /output

## Dijkstra

### Orientation

With `orientation="in"`, the paths go from every vertex to a target vertex.

```python
dijkstra_in = Dijkstra(edges, orientation="in")
dijkstra_in.run(vertex_idx=4)
```

% output

    array([5.5, 4.5, 2.5, 1. , 0. ])

% /output

### Paths

With `path_tracking=True`, the shortest path tree is stored and `get_path` returns the vertices of a path. The array starts with the vertex given to `get_path` and ends with the vertex given to `run`: with the default orientation the path is in reverse order, from the target back to the source, and with `orientation="in"` it is in travel order.

```python
dijkstra_in.run(vertex_idx=4, path_tracking=True)
dijkstra_in.get_path(vertex_idx=0)
```

% output

    array([0, 1, 2, 3, 4], dtype=uint32)

% /output

The tree itself is the `path_links` attribute: the predecessor (`"out"`) or successor (`"in"`) of each vertex.

### Early termination

When only a few path lengths are needed, `termination_nodes` stops the algorithm as soon as these vertices are settled. The result then has one element per termination node, in the given order.

```python
dijkstra.run(vertex_idx=0, termination_nodes=[4, 2])
```

% output

    array([5.5, 3. ])

% /output

### Several sources

`run_many` runs the algorithm from several vertices in parallel threads, `n_jobs` of them (all the CPUs by default). The result has one row per source; with `termination_nodes`, one column per termination node.

```python
dijkstra.run_many([0, 1, 2])
```

% output

    array([[0. , 1. , 3. , 4.5, 5.5],
           [inf, 0. , 2. , 3.5, 4.5],
           [inf, inf, 0. , 1.5, 2.5]])

% /output

With `output="pandas"`, the result is a DataFrame indexed by origin, with one column per destination. With `output="polars"`, it is in long format, one row per pair of vertices:

```python
dijkstra.run_many([0, 1], termination_nodes=[3, 4], output="polars")
```

% output

    shape: (4, 3)
    ┌────────┬─────────────┬─────────────┐
    │ origin ┆ destination ┆ path_length │
    │ ---    ┆ ---         ┆ ---         │
    │ i64    ┆ i64         ┆ f64         │
    ╞════════╪═════════════╪═════════════╡
    │ 0      ┆ 3           ┆ 4.5         │
    │ 0      ┆ 4           ┆ 5.5         │
    │ 1      ┆ 3           ┆ 3.5         │
    │ 1      ┆ 4           ┆ 4.5         │
    └────────┴─────────────┴─────────────┘

% /output

### Output types

`run` returns a NumPy array indexed by vertex identifier by default. `output="pandas"` returns a Series named `path_length` and indexed by `vertex_idx`, `output="polars"` a DataFrame with these two columns. `return_series=True`, from earlier versions, is the same as `output="pandas"`.

### Vertex identifiers

A graph whose largest vertex identifier is 1004 is handled as a graph with 1005 vertices, even if only 5 of them have edges. With `permute=True`, the vertices are renumbered internally with contiguous indices, which saves memory and time when the identifiers are sparse. The arguments and the results keep the original identifiers.

```python
shifted = edges.assign(tail=edges["tail"] + 1000, head=edges["head"] + 1000)

dijkstra_p = Dijkstra(shifted, permute=True)
dijkstra_p.run(vertex_idx=1000, output="pandas")
```

% output

    vertex_idx
    1000    0.0
    1001    1.0
    1002    3.0
    1003    4.5
    1004    5.5
    Name: path_length, dtype: float64

% /output

The Series has the 5 vertices of the graph only, whereas a NumPy array would have 1005 elements. Negative identifiers are also accepted with `permute=True`; an array indexed by vertex cannot hold them, so the output must then be `"pandas"` or `"polars"`.

### Unreachable vertices

The path length of a vertex that cannot be reached is `inf`, or the largest `float64` value (about 1.8e308) with `return_inf=False`.

```python
dijkstra_in.run(vertex_idx=0)
```

% output

    array([ 0., inf, inf, inf, inf])

% /output

### Input checks

Negative and NaN weights always raise an error when the graph is built, because Dijkstra's algorithm is not valid with negative weights.

```python
Dijkstra(edges.assign(weight=-edges["weight"]))
```

% error

    ValueError: edges['weight'] should be nonnegative

% /output

The other checks are optional: with `check_edges=True`, the column names, the data types and the absence of missing values are verified before the graph is built.

```python
Dijkstra(edges.astype({"tail": float}), check_edges=True)
```

% error

    TypeError: edges['tail'] should be of integer type

% /output

### Parallel edges

When several edges have the same tail and head, only the one with the smallest weight is kept.

```python
parallel = pd.DataFrame(
    {"tail": [0, 0, 1], "head": [1, 1, 2], "weight": [5.0, 3.0, 2.0]}
)
Dijkstra(parallel).run(vertex_idx=0)
```

% output

    array([0., 3., 5.])

% /output

## Bellman-Ford

`BellmanFord` has the same interface as `Dijkstra` and accepts negative weights. Its run time grows with the product of the number of vertices and the number of edges, so `Dijkstra` is the better choice when all the weights are non-negative.

```python
from edsger import BellmanFord

edges_neg = pd.DataFrame(
    {
        "tail": [0, 0, 1, 1, 2, 3],
        "head": [1, 2, 2, 3, 3, 4],
        "weight": [1.0, 4.0, -2.0, 5.0, 1.0, 3.0],
    }
)
BellmanFord(edges_neg).run(vertex_idx=0)
```

% output

    array([ 0.,  1., -1.,  0.,  3.])

% /output

Shortest paths are not defined when the graph has a cycle of negative total weight. Such a cycle raises an error, unless the detection is skipped with `detect_negative_cycles=False`; the path lengths are then meaningless.

```python
cycle = pd.DataFrame(
    {"tail": [0, 1, 2], "head": [1, 2, 0], "weight": [1.0, -2.0, -1.0]}
)
BellmanFord(cycle).run(vertex_idx=0)
```

% error

    ValueError: Negative cycle detected in the graph

% /output

## Between two vertices: A* and bidirectional Dijkstra

`AStar` and `BidirectionalDijkstra` compute one shortest path, between a source and a target. They keep their working arrays between runs, so a query costs time in proportion to the part of the graph it explores, not to the size of the graph: on a large road network, a query between nearby vertices takes a fraction of a millisecond, where `Dijkstra.run` has to initialize one value per vertex.

```python
from edsger import BidirectionalDijkstra

bidir = BidirectionalDijkstra(edges)
bidir.run(source=0, target=4, path_tracking=True)
```

% output

    5.5

% /output

`get_path` returns the path of the last run in travel order, from the source to the target.

```python
bidir.get_path()
```

% output

    array([0, 1, 2, 3, 4], dtype=uint32)

% /output

A* is guided by a heuristic: for each vertex, a lower bound of the path length to the target. The search then explores the vertices that look closer to the target first. The heuristic is given to `run` as an array indexed by vertex (or a Series), or computed from vertex coordinates with `set_coordinates`.

```python
import numpy as np

from edsger import AStar

astar = AStar(edges)
heuristic = np.array([5.0, 4.0, 2.0, 1.0, 0.0])  # lower bounds of the distance to 4
astar.run(source=0, target=4, heuristic=heuristic)
```

% output

    5.5

% /output

The path found is the shortest one if the heuristic never overestimates the path length to the target. With `set_coordinates`, the heuristic is a distance between the vertex coordinates (Euclidean, or great-circle with `metric="haversine"`) times a factor computed from the edges so that this holds; see [Routing on a road network](../examples/road_routing.md).
