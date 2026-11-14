# Quick start

## Graph format

A graph is a dataframe with one row per directed edge.

| Column | Type | Content |
|---|---|---|
| `tail` | integer | Start vertex of the edge |
| `head` | integer | End vertex of the edge |
| `weight` | integer or float | Weight of the edge |

These are the default column names. Other names can be given with the `tail`, `head` and `weight` arguments of the classes.

The vertex identifiers are non-negative integers, and they are used as array indices: a result array has one element per integer between 0 and the largest identifier. When the identifiers are large, far apart or negative, see [Vertex identifiers](#vertex-identifiers).

The dataframe can be a pandas dataframe, with NumPy or Arrow data types, or a Polars dataframe. In all cases the columns are converted to NumPy arrays when the graph is built.

```python
import polars as pl

from edsger.path import Dijkstra

edges_pl = pl.DataFrame(
    {"tail": [0, 0, 1], "head": [1, 2, 2], "weight": [1.0, 4.0, 2.0]}
)
Dijkstra(edges_pl).run(vertex_idx=0)
```

    array([0., 1., 3.])

## Dijkstra

The `Dijkstra` class computes shortest paths in a graph with non-negative weights. The graph is built once, when the object is created.

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
```

The `run` method returns the length of the shortest path from a source vertex to every vertex. It can be called any number of times with different sources.

```python
dijkstra.run(vertex_idx=0)
```

    array([0. , 1. , 3. , 4.5, 5.5])

### Orientation

With `orientation="in"`, the paths go from every vertex to a target vertex.

```python
dijkstra_in = Dijkstra(edges, orientation="in")
dijkstra_in.run(vertex_idx=4)
```

    array([5.5, 4.5, 2.5, 1. , 0. ])

### Paths

With `path_tracking=True`, the shortest path tree is stored and `get_path` returns the vertices of a path. The array starts with the vertex given to `get_path` and ends with the vertex given to `run`.

```python
dijkstra.run(vertex_idx=0, path_tracking=True)
dijkstra.get_path(vertex_idx=4)
```

    array([4, 3, 2, 1, 0], dtype=uint32)

So with the default orientation the path is in reverse order, from the target back to the source. With `orientation="in"` it is in travel order.

```python
dijkstra_in.run(vertex_idx=4, path_tracking=True)
dijkstra_in.get_path(vertex_idx=0)
```

    array([0, 1, 2, 3, 4], dtype=uint32)

### Early termination

When only a few path lengths are needed, `termination_nodes` stops the algorithm as soon as these vertices are settled. The result then has one element per termination node, in the given order.

```python
dijkstra.run(vertex_idx=0, termination_nodes=[4, 2])
```

    array([5.5, 3. ])

### Vertex identifiers

Without any option, a graph whose largest vertex identifier is 1004 is handled as a graph with 1005 vertices, even if only 5 of them have edges. With `permute=True`, the vertices are renumbered internally with contiguous indices, which saves memory and time when the identifiers are sparse. The arguments and the results keep the original identifiers.

```python
shifted = edges.assign(tail=edges["tail"] + 1000, head=edges["head"] + 1000)

dijkstra_p = Dijkstra(shifted, permute=True)
dijkstra_p.run(vertex_idx=1000, output="pandas")
```

    vertex_idx
    1000    0.0
    1001    1.0
    1002    3.0
    1003    4.5
    1004    5.5
    Name: path_length, dtype: float64

`output="pandas"` returns a pandas Series indexed by vertex, here with the 5 vertices of the graph only. The default NumPy array would still have 1005 elements. `output="polars"` returns a Polars DataFrame with a `vertex_idx` and a `path_length` column. `return_series=True`, from the earlier versions, is the same as `output="pandas"`.

Negative identifiers are also accepted with `permute=True`. An array indexed by vertex cannot hold them, so the output must then be `"pandas"` or `"polars"`.

### Unreachable vertices

The path length of a vertex that cannot be reached is `inf`.

```python
dijkstra_in.run(vertex_idx=0)
```

    array([ 0., inf, inf, inf, inf])

With `return_inf=False`, the largest `float64` value (about 1.8e308) is returned instead.

### Input checks

Negative and NaN weights always raise an error when the graph is built, because Dijkstra's algorithm is not valid with negative weights.

```python
Dijkstra(edges.assign(weight=-edges["weight"]))
```

    ValueError: edges['weight'] should be nonnegative

The other checks are optional. With `check_edges=True`, the column names, the data types and the absence of missing values are verified before the graph is built.

```python
Dijkstra(edges.astype({"tail": float}), check_edges=True)
```

    TypeError: edges['tail'] should be of integer type

### Parallel edges

When several edges have the same tail and head, only the one with the smallest weight is kept.

```python
parallel = pd.DataFrame(
    {"tail": [0, 0, 1], "head": [1, 1, 2], "weight": [5.0, 3.0, 2.0]}
)
dijkstra_m = Dijkstra(parallel)
dijkstra_m.n_edges, dijkstra_m.run(vertex_idx=0)
```

    (2, array([0., 3., 5.]))

## Bellman-Ford

The `BellmanFord` class has the same interface as `Dijkstra` and accepts negative weights. Its run time grows with the product of the number of vertices and the number of edges, so `Dijkstra` is the better choice when all the weights are non-negative.

```python
from edsger.path import BellmanFord

edges_neg = pd.DataFrame(
    {
        "tail": [0, 0, 1, 1, 2, 3],
        "head": [1, 2, 2, 3, 3, 4],
        "weight": [1.0, 4.0, -2.0, 5.0, 1.0, 3.0],
    }
)
bellman_ford = BellmanFord(edges_neg)
bellman_ford.run(vertex_idx=0)
```

    array([ 0.,  1., -1.,  0.,  3.])

Shortest paths are not defined when the graph has a cycle of negative total weight. Such a cycle raises an error.

```python
cycle = pd.DataFrame(
    {"tail": [0, 1, 2], "head": [1, 2, 0], "weight": [1.0, -2.0, -1.0]}
)
BellmanFord(cycle).run(vertex_idx=0)
```

    ValueError: Negative cycle detected in the graph

The detection can be skipped with `detect_negative_cycles=False`. The returned path lengths are then meaningless if there is a negative cycle.

## Breadth-first search

The `BFS` class ignores the weights and finds the paths with the fewest edges. Its `run` method does not return path lengths but the predecessor of each vertex on such a path.

```python
from edsger.path import BFS

bfs = BFS(edges)
bfs.run(vertex_idx=0)
```

    array([-9999,     0,     0,     2,     2], dtype=int32)

The source vertex and the vertices that cannot be reached have the value -9999, which can be changed with the `sentinel` argument. Paths are retrieved as with the other classes.

```python
bfs.run(vertex_idx=0, path_tracking=True)
bfs.get_path(vertex_idx=4)
```

    array([4, 2, 0], dtype=uint32)

## Hyperpaths

The `HyperpathGenerating` class implements the algorithm of [Spiess and Florian (1989)](https://doi.org/10.1016/0191-2615(89)90034-9) for frequency-based transit networks. Each edge has a travel time and a frequency. At a stop, a traveller boards the first vehicle to come among a set of attractive lines, so the travel time includes an expected waiting time, and the demand is split between the lines in proportion to their frequencies.

| Column | Type | Content |
|---|---|---|
| `tail` | integer | Start vertex of the edge |
| `head` | integer | End vertex of the edge |
| `trav_time` | float | Travel time on the edge |
| `freq` | float | Frequency of the edge, in vehicles per unit of travel time |

In the following network, times are in minutes. A direct line goes from vertex 0 to vertex 2 in 25 minutes, every 12 minutes. Another line goes from 0 to 1 in 7 minutes, every 12 minutes, and a third one from 1 to 2 in 10 minutes, every 6 minutes.

```python
from edsger.path import HyperpathGenerating

transit = pd.DataFrame(
    {
        "tail": [0, 0, 1],
        "head": [2, 1, 2],
        "trav_time": [25.0, 7.0, 10.0],
        "freq": [1 / 12, 1 / 12, 1 / 6],
    }
)
hp = HyperpathGenerating(transit)
hp.run(origin=0, destination=2, volume=100.0)
```

The `run` method returns nothing. The expected travel time from each vertex to the destination is stored in `u_i_vec`.

```python
hp.u_i_vec
```

    array([30., 16.,  0.])

From vertex 1, the expected waiting time is 6 minutes and the ride takes 10 minutes. From vertex 0, both lines are attractive: the first vehicle comes after 6 minutes on average, and half of the travellers end up on each line, with 25 and 23 minutes left to the destination. The edge volumes are stored in the `volume` column of the `_edges` dataframe.

```python
hp._edges[["tail", "head", "volume"]]
```

       tail  head  volume
    0     0     2    50.0
    1     0     1    50.0
    2     1     2    50.0

Several origins can be assigned in one run, with one volume per origin.

```python
hp.run(origin=[0, 1], destination=2, volume=[100.0, 50.0])
hp._edges[["tail", "head", "volume"]]
```

       tail  head  volume
    0     0     2    50.0
    1     0     1    50.0
    2     1     2   100.0
