# Quick Start Guide

Welcome to our Python library for graph algorithms. So far, the library only includes Dijkstra's algorithm but we should add a range of common path algorithms later. It is also open-source and easy to integrate with other Python libraries.


## Graph Data Format

Edsger expects graph data as a pandas DataFrame with the following structure:

| Column | Type    | Description                          |
|--------|---------|--------------------------------------|
| tail   | int     | Source vertex ID                     |
| head   | int     | Destination vertex ID                |
| weight | float   | Edge weight (must be non-negative)   |

Example:
```python
import pandas as pd

edges = pd.DataFrame({
    'tail': [0, 0, 1, 2],
    'head': [1, 2, 2, 3],
    'weight': [1.0, 4.0, 2.0, 1.0]
})
edsges
```

|    |   tail |   head |   weight |
|---:|-------:|-------:|---------:|
|  0 |      0 |      1 |        1 |
|  1 |      0 |      2 |        4 |
|  2 |      1 |      2 |        2 |
|  3 |      2 |      3 |        1 |


## Dijkstra's Algorithm

To use Dijkstra's algorithm, you can import the `Dijkstra` class from the `path` module. The function takes a graph and a source node as input, and returns the shortest path from the source node to all other nodes in the graph.

```python
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

It is also possible to use a graph with different column names for the tail, head and weight values. The column names can be specified using the `tail`, `head` and `weight` arguments:

```python
other_edges = pd.DataFrame({
    'from': [0, 0, 1, 2, 2, 3],
    'to': [1, 2, 2, 3, 4, 4],
    'travel_time': [1, 4, 2, 1.5, 3, 1]
})
other_dijkstra = Dijkstra(other_edges, tail='from', head='to', weight='travel_time')
```

### Orientation

The `orientation` argument (a string with a default value of `'out'`) specifies the orientation of the algorithm. It can be either `'out'` for single source shortest paths or `'in'` for single target shortest path.

```python
dijkstra = Dijkstra(edges, orientation='in')

# Run the algorithm to a target vertex
shortest_paths = dijkstra.run(vertex_idx=0)
print("Shortest paths:", shortest_paths)
```

    Shortest paths: [ 0. inf inf inf inf]

### Run Multiple Times

Once the Dijkstra is instantiated with a given graph and orientation, the `run` method can be called multiple times with different source vertices.

```python
# Run the algorithm to another target vertex
shortest_paths = dijkstra.run(vertex_idx=4)
print("Shortest paths:", shortest_paths)
```

    Shortest paths: [5.5 4.5 2.5 1.  0. ]

### Check Edges

The `check_edges` argument (a boolean with a default value of `False`) validates the given graph. When set to `True`, it ensures the DataFrame is well-formed by:

- Checking for the presence of required columns for edge tail, head and weight values
- Verifying that the data types are correct (integer for tail and head, integer or float for weight)
- Ensuring there are no missing or invalid values (e.g. negative weights)

If any of these checks fail, an appropriate error is raised.

```python
invalid_edges = pd.DataFrame({
    'tail': [0, 0, 1, 2, 2, 3],
    'head': [1, 2, 2, 3, 4, 4],
    'weight': [1, 4, 2, -1, 3, 1]
})
dijkstra = Dijkstra(invalid_edges, check_edges=True)
```

    ValueError: edges['weight'] should be nonnegative

### Permute

Finally, the `permute` argument (boolean with a default value of `False`) allows to permute the IDs of the nodes. If set to `True`, the node IDs will be reindexed to start from 0 and be contiguous for the inner computations, and the output will be reindexed to the original IDs, loading the same result as if the IDs were not permuted. The permutation may save memory and computation time for large graphs, for example if a significant ratio of the nodes are not actually used in the graph.

```python
SHIFT = 1000

shifted_edges = pd.DataFrame({
    'tail': [0, 0, 1, 2, 2, 3],
    'head': [1, 2, 2, 3, 4, 4],
    'weight': [1, 4, 2, 1.5, 3, 1]
})
shifted_edges["tail"] += SHIFT
shifted_edges["head"] += SHIFT
shifted_edges.head(3)
```

|    |   tail |   head |   weight |
|---:|-------:|-------:|---------:|
|  0 |   1000 |   1001 |      1.0 |
|  1 |   1000 |   1002 |      4.0 |
|  2 |   1001 |   1002 |      2.0 |



```python
dijkstra = Dijkstra(shifted_edges, permute=True)
shortest_paths = dijkstra.run(vertex_idx=0 + SHIFT)
print("Shortest paths:", shortest_paths)
```

    Shortest paths: [inf inf inf ... 3.  4.5 5.5]

```python
shortest_paths[-5:]
```

    array([0. , 1. , 3. , 4.5, 5.5])


### Run Method Options

The `run` method can take the following arguments besides the source/target vertex index:

- `path_tracking` : bool, optional (default=False)

Whether to track the shortest path(s) from/to the source/target vertex to all other vertices in the graph.

```python
dijkstra = Dijkstra(edges)
shortest_paths = dijkstra.run(vertex_idx=0, path_tracking=True)
dijkstra.get_path(vertex_idx=4)
```
    
    array([4, 3, 2, 1, 0], dtype=uint32)

```python
dijkstra.get_path(vertex_idx=0)
```

    array([0], dtype=uint32)

The path is returned as an array of vertex indices. This is an ordered list of vertices from the source to the target vertex if `orientation` is `'in'`, and from the target to the source vertex if `orientation` is `'out'`. Both the source and target vertices are included in the path.

- `return_inf` : bool, optional (default=True)
    
Whether to return path lengths as infinity (np.inf) when no path exists.

```python
dijkstra = Dijkstra(edges, orientation='in')
shortest_paths = dijkstra.run(vertex_idx=0, return_inf=False)
shortest_paths
```

    array([0.00000000e+000, 1.79769313e+308, 1.79769313e+308, 1.79769313e+308,
       1.79769313e+308])

The value 1.79769313e+308 actually used in the code is the largest number that can be represented in the floating point format (`np.float64`).

- `return_series` : bool, optional (default=False)

Instead of returning a NumPy array, the `run` method may return a Pandas Series object with the path lengths as values and the vertex indices as the index.

```python
shortest_paths = dijkstra.run(vertex_idx=4, return_series=True)
shortest_paths
```

|   vertex_idx |   path_length |
|-------------:|--------------:|
|            0 |           5.5 |
|            1 |           4.5 |
|            2 |           2.5 |
|            3 |           1.0 |
|            4 |           0.0 |


- `heap_length_ratio` : float, optional (default=1.0)
    
This is an experimental parameter that controls the size of the heap used in the algorithm. The heap is a static array that is used to store the vertices that may be visited next. A value of 1.0 means that the heap is the same size as the number of vertices, so there is no risk of overflow. Be aware that there is no guarantee that the algorithm will work with a heap length ratio smaller that 1. The lowest ratio that works for a given graph depends on the graph structure and the source vertex. For a rather sparse graph, a small ratio may work, but for a dense graph, a ratio of 1.0 is required.

## Next Steps

- Explore the [API Reference](api.md) for more detailed information
- Learn about [Contributing](contributing.md) to Edsger