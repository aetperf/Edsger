# Coming from NetworkX or SciPy

Edsger works on a graph built once from a dataframe of edges, and runs its algorithms any number of times on it. This page maps the usual functions of NetworkX and `scipy.sparse.csgraph` to Edsger.

## Building the graph

| NetworkX / SciPy | Edsger |
|---|---|
| `nx.from_pandas_edgelist(df, "u", "v", edge_attr="w", create_using=nx.DiGraph)` | `Dijkstra(df, tail="u", head="v", weight="w")` |
| `csr_matrix((w, (u, v)), shape=(n, n))` | `Dijkstra(pd.DataFrame({"tail": u, "head": v, "weight": w}))` |
| `nx.Graph` (undirected) | add each edge in both directions, see below |
| Arbitrary hashable nodes | integer identifiers; `permute=True` when they are sparse or negative |

The graphs are directed. For an undirected graph, give each edge in both directions:

```python
import pandas as pd

from edsger import Dijkstra

undirected = pd.DataFrame({"tail": [0, 1], "head": [1, 2], "weight": [1.0, 2.0]})
both_ways = pd.concat(
    [undirected, undirected.rename(columns={"tail": "head", "head": "tail"})]
)
Dijkstra(both_ways).run(vertex_idx=2)
```

% output

    array([3., 2., 0.])

% /output

`MinimumSpanningTree` and the weakly connected components already ignore the edge directions.

Some differences with NetworkX and SciPy:

- Parallel edges are merged: between two vertices, only the edge with the smallest weight is kept (NetworkX `MultiDiGraph` keeps them all). SciPy sparse matrices sum the duplicate entries.
- A zero weight is a valid edge, whereas the zeros of a dense matrix mean "no edge" in SciPy.
- With the default `permute=False`, the results are arrays indexed by vertex identifier, from 0 to the largest one, like the SciPy results. `output="pandas"` or `output="polars"` labels them by vertex, like the dictionaries of NetworkX.

## Shortest paths

| NetworkX | SciPy | Edsger |
|---|---|---|
| `nx.single_source_dijkstra_path_length(G, s)` | `dijkstra(A, indices=s)` | `Dijkstra(edges).run(s)` |
| `nx.single_source_dijkstra(G, s)` (with paths) | `dijkstra(A, indices=s, return_predecessors=True)` | `run(s, path_tracking=True)`, then `get_path(t)` or `path_links` |
| `nx.single_source_dijkstra_path_length(G.reverse(), t)` | `dijkstra(A.T, indices=t)` | `Dijkstra(edges, orientation="in").run(t)` |
| `nx.dijkstra_path_length(G, s, t)` | | `BidirectionalDijkstra(edges).run(s, t)`, or `Dijkstra.run(s, termination_nodes=[t])` |
| `nx.astar_path_length(G, s, t, heuristic)` | | `AStar(edges).run(s, t, heuristic=h)` |
| `dict(nx.all_pairs_dijkstra_path_length(G))` | `dijkstra(A)` | `Dijkstra(edges).run_many(vertices)` |
| `nx.bellman_ford_path_length(G, s, t)` | `bellman_ford(A, indices=s)` | `BellmanFord(edges).run(s)` |
| `nx.negative_edge_cycle(G)` | `NegativeCycleError` | `ValueError` from `BellmanFord.run`, `has_negative_cycle()` |

The paths of `get_path` go from the given vertex back to the vertex of `run`: reverse it to get the path in travel order with the default orientation. `AStar` and `BidirectionalDijkstra` return their path in travel order.

## Traversal and components

| NetworkX | SciPy | Edsger |
|---|---|---|
| `nx.bfs_predecessors(G, s)` | `breadth_first_order(A, s)` | `BFS(edges).run(s)` |
| `nx.dfs_preorder_nodes(G, s)` | `depth_first_order(A, s)` | `DFS(edges).run(s)` and `get_order()` |
| `nx.weakly_connected_components(G)` | `connected_components(A, connection="weak")` | `ConnectedComponents(edges).run("weak")` |
| `nx.strongly_connected_components(G)` | `connected_components(A, connection="strong")` | `ConnectedComponents(edges).run("strong")` |

`ConnectedComponents.run` returns a label per vertex, like SciPy, rather than sets of vertices. The labels are numbered in increasing order of the smallest vertex of each component.

## Spanning trees and centrality

| NetworkX | SciPy | Edsger |
|---|---|---|
| `nx.minimum_spanning_tree(G)` | `minimum_spanning_tree(A)` | `MinimumSpanningTree(edges).run()` |
| `nx.pagerank(G, alpha)` | | `Centrality(edges).pagerank(damping=alpha)` |
| `nx.hits(G)` | | `Centrality(edges).hits()` |
| `nx.katz_centrality(G, alpha, beta)` | | `Centrality(edges).katz(alpha, beta)` |

The centrality measures have the normalizations of NetworkX. Unlike NetworkX, the PageRank tolerance is not multiplied by the number of vertices: the default `tol=1e-6` gives the same accuracy on large and small graphs.
