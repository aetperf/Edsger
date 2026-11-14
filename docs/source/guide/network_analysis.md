# Spanning trees and centrality

| Class | Problem | Methods |
|---|---|---|
| `MinimumSpanningTree` | Minimum spanning forest of the graph seen as undirected | `run` |
| `Centrality` | Importance of the vertices | `pagerank`, `hits`, `katz` |

## Minimum spanning tree

`MinimumSpanningTree` treats the graph as undirected: an edge connects its two vertices in both directions. `run` returns the edges of a minimum spanning forest, one tree per connected component, with the original tail, head and weight of each edge.

```python
import pandas as pd

from edsger import MinimumSpanningTree

edges = pd.DataFrame(
    {
        "tail": [0, 0, 1, 1, 2, 3, 5],
        "head": [1, 2, 2, 3, 3, 4, 6],
        "weight": [4.0, 1.0, 2.0, 5.0, 8.0, 3.0, 1.0],
    }
)
mst = MinimumSpanningTree(edges)
mst.run()
```

% output

       tail  head  weight
    0     0     2     1.0
    1     5     6     1.0
    2     1     2     2.0
    3     3     4     3.0
    4     1     3     5.0

% /output

```python
mst.total_weight, mst.n_trees
```

% output

    (12.0, 2)

% /output

The forest is returned as a pandas DataFrame by default, a Polars DataFrame with `output="polars"`, or a tuple of NumPy arrays `(tails, heads, weights)` with `output="numpy"`. When two vertices are joined by several edges, in either direction, only one of minimum weight can be selected. Negative weights are accepted; with `weight=None`, all the edges weigh 1. The algorithm is Kruskal's, with a union-find structure.

## Centrality

`Centrality` builds the graph once and computes several measures on it, by power iteration. The edges are unweighted unless a `weight` column is given.

```python
from edsger import Centrality

links = pd.DataFrame(
    {
        "tail": [0, 1, 2, 2, 3, 4, 4],
        "head": [1, 2, 0, 3, 2, 2, 3],
    }
)
centrality = Centrality(links)
centrality.pagerank()
```

% output

    array([0.1917835, 0.1930161, 0.3806669, 0.2045335, 0.03     ])

% /output

The PageRank is the stationary distribution of a random walk that follows an outgoing edge with probability `damping` (0.85), and otherwise jumps to a random vertex, drawn from the `personalization` distribution if one is given. It sums to 1. From a vertex without outgoing edge, the walk always jumps.

`hits` returns the hub and authority scores of the vertices: a good hub points to good authorities, a good authority is pointed to by good hubs.

```python
hubs, authorities = centrality.hits()
authorities.round(4)
```

% output

    array([0.134, 0.   , 0.5  , 0.366, 0.   ])

% /output

`katz` gives each vertex the weighted number of walks that end at it, a walk of length k counting for `alpha` to the power k, plus a constant `beta`:

```python
centrality.katz(alpha=0.1)
```

% output

    array([0.43380133, 0.42610049, 0.51080976, 0.47207337, 0.3827204 ])

% /output

The normalizations are the ones of NetworkX: the PageRank and the HITS scores sum to 1, the Katz centrality has a unit Euclidean norm (`normalized=True`). The iterations stop when the change of the scores is below `tol`, and raise a `ConvergenceError` if this does not happen within `max_iter` iterations. Unlike NetworkX, the PageRank tolerance is not multiplied by the number of vertices, so that the accuracy of the result does not depend on the size of the graph. The Katz iterations only converge if `alpha` is smaller than the inverse of the largest eigenvalue of the adjacency matrix.

The results are labelled by vertex with `output="pandas"` or `output="polars"`, as for the other classes.
