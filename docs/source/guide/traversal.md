# Traversal and components

| Class | Problem | Result |
|---|---|---|
| `BFS` | Paths with the fewest edges | Predecessor of each vertex |
| `DFS` | Depth-first search tree | Predecessor of each vertex, discovery order |
| `ConnectedComponents` | Weakly or strongly connected components | Component of each vertex |

The edge weights are not used by these classes: a weight column, if there is one, is ignored. The examples use the following graph.

```python
import pandas as pd

from edsger import BFS, DFS, ConnectedComponents

edges = pd.DataFrame(
    {
        "tail": [0, 0, 1, 2, 2, 3],
        "head": [1, 2, 2, 3, 4, 4],
    }
)
```

## Breadth-first search

`BFS.run` returns the predecessor of each vertex on a path with the fewest edges from the source (`orientation="out"`), or its successor on such a path to the target (`orientation="in"`).

```python
bfs = BFS(edges)
bfs.run(vertex_idx=0)
```

% output

    array([-9999,     0,     0,     2,     2], dtype=int32)

% /output

The source vertex and the vertices that cannot be reached get the value -9999, which can be changed with the `sentinel` argument (`bfs.UNREACHABLE`). Paths are retrieved as with `Dijkstra`, from the given vertex back to the source:

```python
bfs.run(vertex_idx=0, path_tracking=True)
bfs.get_path(vertex_idx=4)
```

% output

    array([4, 2, 0], dtype=uint32)

% /output

## Depth-first search

`DFS` has the same interface. The search goes as deep as possible before backtracking; the neighbours of a vertex are visited in increasing order of their identifiers, as in `scipy.sparse.csgraph.depth_first_order`. `get_order` returns the vertices in the order they were discovered.

```python
dfs = DFS(edges)
dfs.run(vertex_idx=0)
```

% output

    array([-9999,     0,     1,     2,     3], dtype=int32)

% /output

```python
dfs.get_order()
```

% output

    array([0, 1, 2, 3, 4], dtype=uint32)

% /output

The search is iterative, with an explicit stack: it is not limited by the depth of the Python or C call stack, on graphs with millions of vertices.

## Connected components

Two vertices are in the same weakly connected component if they are connected when the edge directions are ignored, and in the same strongly connected component if there is a path from each one to the other.

```python
graph = pd.DataFrame(
    {
        "tail": [0, 1, 2, 3, 4, 5],
        "head": [1, 2, 0, 4, 3, 6],
    }
)
cc = ConnectedComponents(graph)
cc.run(connection="weak")
```

% output

    array([0, 0, 0, 1, 1, 2, 2], dtype=int32)

% /output

```python
cc.run(connection="strong"), cc.n_components
```

% output

    (array([0, 0, 0, 1, 1, 2, 3], dtype=int32), 4)

% /output

The components are numbered from 0 in increasing order of their smallest vertex, so the labels do not depend on the order of the edges. The weakly connected components are computed with a union-find structure, the strongly connected ones with Pearce's iterative variant of Tarjan's algorithm.

Without `permute`, the integers between 0 and the largest identifier that are not vertices get the label -1 in the array, and are not counted in `n_components`.

## Output types

As for the shortest paths, `run` returns a NumPy array indexed by vertex identifier by default, and `output="pandas"` or `output="polars"` labels the results by vertex:

```python
cc.run(connection="strong", output="pandas")
```

% output

    vertex_idx
    0    0
    1    0
    2    0
    3    1
    4    1
    5    2
    6    3
    Name: component, dtype: int32

% /output
