# API Reference

This section provides detailed documentation for all public classes and functions in Edsger.

## edsger.path Module

### Dijkstra Class

The main class for performing Dijkstra's shortest path algorithm.

#### Constructor Parameters

- `edges`: pandas.DataFrame containing the graph edges
- `tail`: Column name for edge source nodes (default: 'tail')
- `head`: Column name for edge destination nodes (default: 'head')
- `weight`: Column name for edge weights (default: 'weight')
- `orientation`: Either 'out' (single-source) or 'in' (single-target) (default: 'out')
- `check_edges`: Whether to validate edge data (default: False)
- `permute`: Whether to optimize node indexing (default: False)

#### Methods

##### run

```python
def run(self, vertex_idx, path_tracking=False, return_inf=True, 
        return_series=False, heap_length_ratio=1.0)
```

Runs the shortest path algorithm from/to the specified vertex.

**Parameters:**
- `vertex_idx`: Source/target vertex index
- `path_tracking`: Whether to track paths for reconstruction (default: False)
- `return_inf`: Return infinity for unreachable vertices (default: True)
- `return_series`: Return results as pandas Series (default: False)
- `heap_length_ratio`: Heap size as fraction of vertices (default: 1.0)

**Returns:**
- Array or Series of shortest path lengths

##### get_path

```python
def get_path(self, vertex_idx)
```

Reconstructs the shortest path to/from a vertex (requires `path_tracking=True`).

**Parameters:**
- `vertex_idx`: Destination/source vertex index

**Returns:**
- Array of vertex indices forming the path

##### get_vertices

```python
def get_vertices(self)
```

Returns all vertices in the graph.

**Returns:**
- Array of vertex indices

### Examples

#### Basic Usage

```python
from edsger.path import Dijkstra
import pandas as pd

# Create a graph
edges = pd.DataFrame({
    'tail': [0, 0, 1],
    'head': [1, 2, 2],
    'weight': [1, 4, 2]
})

# Initialize Dijkstra
dijkstra = Dijkstra(edges)

# Find shortest paths from vertex 0
paths = dijkstra.run(vertex_idx=0)
```

#### With Path Tracking

```python
# Enable path tracking
paths = dijkstra.run(vertex_idx=0, path_tracking=True)

# Get the actual path to vertex 2
path = dijkstra.get_path(vertex_idx=2)
```

#### Custom Parameters

```python
# Create with custom settings
dijkstra = Dijkstra(
    edges,
    orientation='in',
    check_edges=True,
    permute=True
)

# Run with custom parameters
paths = dijkstra.run(
    vertex_idx=2,
    return_series=True,
    heap_length_ratio=0.5
)
```

### BellmanFord Class

The class for performing Bellman-Ford shortest path algorithm, which handles graphs with negative edge weights and detects negative cycles.

#### Constructor Parameters

- `edges`: pandas.DataFrame containing the graph edges
- `tail`: Column name for edge source nodes (default: 'tail')
- `head`: Column name for edge destination nodes (default: 'head')
- `weight`: Column name for edge weights (default: 'weight')
- `orientation`: Either 'out' (single-source) or 'in' (single-target) (default: 'out')
- `check_edges`: Whether to validate edge data (default: False)
- `permute`: Whether to optimize node indexing (default: False)

#### Methods

##### run

```python
def run(self, vertex_idx, path_tracking=False, return_inf=True, 
        return_series=False, detect_negative_cycles=True)
```

Runs the Bellman-Ford algorithm from/to the specified vertex.

**Parameters:**
- `vertex_idx`: Source/target vertex index
- `path_tracking`: Whether to track paths for reconstruction (default: False)
- `return_inf`: Return infinity for unreachable vertices (default: True)
- `return_series`: Return results as pandas Series (default: False)
- `detect_negative_cycles`: Whether to detect negative cycles (default: True)

**Returns:**
- Array or Series of shortest path lengths

**Raises:**
- `ValueError`: If a negative cycle is detected and `detect_negative_cycles=True`

##### get_path

```python
def get_path(self, vertex_idx)
```

Reconstructs the shortest path to/from a vertex (requires `path_tracking=True`).

**Parameters:**
- `vertex_idx`: Destination/source vertex index

**Returns:**
- Array of vertex indices forming the path

##### get_vertices

```python
def get_vertices(self)
```

Returns all vertices in the graph.

**Returns:**
- Array of vertex indices

#### Examples

##### Basic Usage with Negative Weights

```python
from edsger.path import BellmanFord
import pandas as pd

# Create a graph with negative weights
edges = pd.DataFrame({
    'tail': [0, 0, 1, 1, 2, 3],
    'head': [1, 2, 2, 3, 3, 4],
    'weight': [1, 4, -2, 5, 1, 3]  # Note the negative weight
})

# Initialize Bellman-Ford
bf = BellmanFord(edges)

# Find shortest paths from vertex 0
paths = bf.run(vertex_idx=0)
print(paths)  # [ 0.  1. -1.  0.  3.]
```

##### Negative Cycle Detection

```python
# Create a graph with a negative cycle
edges_cycle = pd.DataFrame({
    'tail': [0, 1, 2],
    'head': [1, 2, 0],
    'weight': [1, -2, -1]  # Cycle 0→1→2→0 has total weight -2
})

bf_cycle = BellmanFord(edges_cycle)
try:
    bf_cycle.run(vertex_idx=0)
except ValueError as e:
    print(f"Error: {e}")  # Error: Negative cycle detected in the graph
```

##### Path Tracking with Negative Weights

```python
# Enable path tracking
paths = bf.run(vertex_idx=0, path_tracking=True)

# Get the actual path to vertex 2 (using negative weight path)
path = bf.get_path(vertex_idx=2)
print(path)  # Path using the negative weight edge
```

##### Performance: Disabling Cycle Detection

```python
# For performance when you know there are no negative cycles
paths = bf.run(vertex_idx=0, detect_negative_cycles=False)
```

## Algorithm Comparison

| Feature | Dijkstra | BellmanFord |
|---------|----------|-------------|
| **Negative weights** | ❌ No | ✅ Yes |
| **Negative cycle detection** | ❌ No | ✅ Yes |
| **Time complexity** | O((V + E) log V) | O(VE) |
| **Space complexity** | O(V) | O(V) |
| **Use case** | Positive weights only | Any weights, cycle detection |
| **Performance** | Faster | Slower but more versatile |
