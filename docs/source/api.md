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
