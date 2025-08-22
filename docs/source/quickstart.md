# Quick Start Guide

Welcome to our Python library for graph algorithms. The library includes both Dijkstra's and Bellman-Ford's algorithms, with plans to add more common path algorithms later. It is also open-source and easy to integrate with other Python libraries.


## Graph Data Format

Edsger expects graph data as a pandas DataFrame with the following structure:

| Column | Type    | Description                                    |
|--------|---------|------------------------------------------------|
| tail   | int     | Source vertex ID                               |
| head   | int     | Destination vertex ID                          |
| weight | float   | Edge weight (non-negative for Dijkstra, can be negative for Bellman-Ford) |

Example:
```python
import pandas as pd

edges = pd.DataFrame({
    'tail': [0, 0, 1, 2],
    'head': [1, 2, 2, 3],
    'weight': [1.0, 4.0, 2.0, 1.0]
})
edges
```

|    |   tail |   head |   weight |
|---:|-------:|-------:|---------:|
|  0 |      0 |      1 |        1 |
|  1 |      0 |      2 |        4 |
|  2 |      1 |      2 |        2 |
|  3 |      2 |      3 |        1 |


Note that it is also possible to use a graph with different column names for the tail, head and weight values, but we need then to specify the name mapping, as described in the following.

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

The column names can be specified using the `tail`, `head` and `weight` arguments:

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

### Early Termination

Early termination is a performance optimization feature that allows Dijkstra's algorithm to stop computing once specific target nodes (termination nodes) have been reached. This can significantly reduce computation time when you only need shortest paths to a subset of vertices in the graph.

When using early termination, the algorithm will:
1. Stop as soon as all specified termination nodes have been visited
2. Return **only** the path lengths to the termination nodes (not all vertices)
3. Return results in the same order as the termination nodes were specified

#### Basic Early Termination Example

```python
import pandas as pd
from edsger.path import Dijkstra

# Create a sample graph
edges = pd.DataFrame({
    "tail": [0, 0, 0, 1, 1, 2, 2, 3, 3, 4],
    "head": [1, 2, 3, 2, 4, 3, 5, 4, 5, 5],
    "weight": [1.0, 4.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0, 1.0, 1.0],
})
```

**Without early termination** (computes paths to all vertices):
```python
dijkstra = Dijkstra(edges, orientation="out")
distances = dijkstra.run(vertex_idx=0)
print("All distances:", distances)
```
    All distances: [0. 1. 2. 2. 3. 3.]

**With early termination** (computes paths only to specified nodes):
```python
# Only compute paths to nodes 3 and 5
termination_nodes = [3, 5]
distances = dijkstra.run(vertex_idx=0, termination_nodes=termination_nodes)
print("Distances to termination nodes:", distances)
print("Shape of result:", distances.shape)
```
    Distances to termination nodes: [2. 3.]
    Shape of result: (2,)

Notice that:
- The result array has length 2 (same as number of termination nodes)
- `distances[0] = 2.0` is the shortest path length from vertex 0 to vertex 3
- `distances[1] = 3.0` is the shortest path length from vertex 0 to vertex 5

#### Early Termination with Path Tracking

Early termination also works with path tracking enabled:

```python
dijkstra = Dijkstra(edges, orientation="out")
distances = dijkstra.run(vertex_idx=0, termination_nodes=[3, 5], path_tracking=True)
print("Distances:", distances)

# Get paths to termination nodes
path_to_3 = dijkstra.get_path(vertex_idx=3)
path_to_5 = dijkstra.get_path(vertex_idx=5)
print("Path to vertex 3:", path_to_3)
print("Path to vertex 5:", path_to_5)
```
    Distances: [2. 3.]
    Path to vertex 3: [3 2 1 0]
    Path to vertex 5: [5 3 2 1 0]

#### Important Notes

1. **Return Array Size**: With early termination, the returned array size equals the number of termination nodes, not the total number of vertices in the graph.

2. **Order Preservation**: Results are returned in the same order as the termination nodes are specified:
   ```python
   # Termination nodes [3, 5] → results [distance_to_3, distance_to_5]
   # Termination nodes [5, 3] → results [distance_to_5, distance_to_3]
   ```

3. **Orientation Support**: Early termination works with both orientations:
   ```python
   # Single-source shortest paths (from source to termination nodes)
   dijkstra = Dijkstra(edges, orientation="out")
   distances = dijkstra.run(vertex_idx=0, termination_nodes=[3, 5])
   
   # Single-target shortest paths (from termination nodes to target)
   dijkstra = Dijkstra(edges, orientation="in") 
   distances = dijkstra.run(vertex_idx=5, termination_nodes=[0, 2])
   ```

4. **Unreachable Nodes**: If a termination node is unreachable, its distance will be infinity:
   ```python
   # If node 10 is unreachable from node 0
   distances = dijkstra.run(vertex_idx=0, termination_nodes=[3, 10])
   # Result: [2.0, inf]
   ```

### Run Method Options

The `run` method can take the following arguments besides the source/target vertex index:

- `termination_nodes` : list or array-like, optional (default=None)

A list or array of vertex indices where the algorithm should stop early. When specified, the algorithm will terminate as soon as all termination nodes have been reached, and will return only the path lengths to these nodes in the same order they were specified. This can provide significant performance improvements when you only need paths to a subset of vertices.

```python
dijkstra = Dijkstra(edges)
# Get distances only to nodes 2 and 4
distances = dijkstra.run(vertex_idx=0, termination_nodes=[2, 4])
print("Distances to nodes 2 and 4:", distances)
```
    Distances to nodes 2 and 4: [1.5 3.5]

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

**Note**: When using `termination_nodes` with `path_tracking=True`, you can still retrieve paths to any vertex that was reached during the computation using `get_path()`, even if it wasn't in the termination nodes list.

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

## Bellman-Ford Algorithm

The Bellman-Ford algorithm is designed for graphs that may contain negative edge weights and can detect negative cycles. Unlike Dijkstra's algorithm, Bellman-Ford can handle a broader class of problems but with a higher computational cost.

### Basic Usage

```python
from edsger.path import BellmanFord

# Create a graph with negative weights
edges_negative = pd.DataFrame({
    'tail': [0, 0, 1, 1, 2, 3],
    'head': [1, 2, 2, 3, 3, 4],
    'weight': [1, 4, -2, 5, 1, 3]  # Note the negative weight
})

# Initialize the Bellman-Ford object
bf = BellmanFord(edges_negative)

# Run the algorithm from a source vertex
shortest_paths = bf.run(vertex_idx=0)
print("Shortest paths:", shortest_paths)
```

    Shortest paths: [ 0.  1. -1.  0.  3.]

### Negative Weights and Optimal Paths

The power of Bellman-Ford becomes evident when dealing with negative weights. In the example above, the shortest path from vertex 0 to vertex 2 has length -1 (going 0→1→2 with weights 1 + (-2) = -1), which is shorter than the direct path 0→2 with weight 4.

### Negative Cycle Detection

One of Bellman-Ford's key features is detecting negative cycles, which make shortest path problems ill-defined:

```python
# Create a graph with a negative cycle
edges_cycle = pd.DataFrame({
    'tail': [0, 1, 2],
    'head': [1, 2, 0],
    'weight': [1, -2, -1]  # Cycle 0→1→2→0 has total weight -2
})

bf_cycle = BellmanFord(edges_cycle)
try:
    shortest_paths = bf_cycle.run(vertex_idx=0)
except ValueError as e:
    print("Error:", e)
```

    Error: Negative cycle detected in the graph

### Disabling Negative Cycle Detection

For performance reasons, you can disable negative cycle detection if you're confident your graph doesn't contain negative cycles:

```python
bf = BellmanFord(edges_negative)
# Skip negative cycle detection for better performance
shortest_paths = bf.run(vertex_idx=0, detect_negative_cycles=False)
```

### Orientation

Like Dijkstra, Bellman-Ford supports both orientations:

```python
# Single-source shortest paths (from source to all vertices)
bf_out = BellmanFord(edges_negative, orientation='out')
distances_out = bf_out.run(vertex_idx=0)

# Single-target shortest paths (from all vertices to target)
bf_in = BellmanFord(edges_negative, orientation='in')
distances_in = bf_in.run(vertex_idx=4)
```

### Path Tracking

Bellman-Ford also supports path reconstruction:

```python
bf = BellmanFord(edges_negative)
shortest_paths = bf.run(vertex_idx=0, path_tracking=True)

# Get the actual path to vertex 3
path = bf.get_path(vertex_idx=3)
print("Path to vertex 3:", path)
```

    Path to vertex 3: [3 2 1 0]

### When to Use Bellman-Ford vs Dijkstra

**Use Bellman-Ford when:**
- Your graph contains negative edge weights
- You need to detect negative cycles
- Working with financial networks, arbitrage detection, or differential constraints
- Performance is less critical than correctness

**Use Dijkstra when:**
- All edge weights are non-negative (e.g., distances, travel times, costs)
- You need the fastest possible performance (O((V+E)logV) vs O(VE))
- Working with road networks, shortest distance problems

### Performance Considerations

Bellman-Ford has O(VE) time complexity compared to Dijkstra's O((V+E)logV). For large graphs with only positive weights, Dijkstra is significantly faster. However, the performance difference may be acceptable for smaller graphs or when negative weights are essential to your problem.

## Next Steps

- Explore the [API Reference](api.md) for more detailed information
- Learn about [Contributing](contributing.md) to Edsger