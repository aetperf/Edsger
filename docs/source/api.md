# API Reference

This section provides detailed documentation for all public classes and functions in Edsger.

## edsger.path Module

### Dijkstra Class

The main class for performing Dijkstra's shortest path algorithm on directed graphs.

#### Constructor Parameters

- `edges`: pandas.DataFrame containing the directed graph edges
- `tail`: Column name for edge source nodes (default: 'tail')
- `head`: Column name for edge destination nodes (default: 'head')
- `weight`: Column name for edge weights (default: 'weight')
- `orientation`: Either 'out' (single-source) or 'in' (single-target) (default: 'out')
- `check_edges`: Whether to validate edge data (default: False)
- `permute`: Whether to optimize node indexing (default: False)
- `verbose`: Whether to print messages about parallel edge removal (default: False)

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

# Create a directed graph
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

The class for performing Bellman-Ford shortest path algorithm on directed graphs, which handles directed graphs with negative edge weights and detects negative cycles.

#### Constructor Parameters

- `edges`: pandas.DataFrame containing the directed graph edges
- `tail`: Column name for edge source nodes (default: 'tail')
- `head`: Column name for edge destination nodes (default: 'head')
- `weight`: Column name for edge weights (default: 'weight')
- `orientation`: Either 'out' (single-source) or 'in' (single-target) (default: 'out')
- `check_edges`: Whether to validate edge data (default: False)
- `permute`: Whether to optimize node indexing (default: False)
- `verbose`: Whether to print messages about parallel edge removal (default: False)

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

# Create a directed graph with negative weights
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
# Create a directed graph with a negative cycle
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

### BFS Class

The class for performing Breadth-First Search on directed graphs, finding shortest paths based on minimum number of edges rather than weighted distances.

#### Constructor Parameters

- `edges`: pandas.DataFrame containing the directed graph edges
- `tail`: Column name for edge source nodes (default: 'tail')
- `head`: Column name for edge destination nodes (default: 'head')
- `orientation`: Either 'out' (single-source) or 'in' (single-target) (default: 'out')
- `check_edges`: Whether to validate edge data (default: False)
- `permute`: Whether to optimize node indexing (default: False)
- `verbose`: Whether to print messages about parallel edge removal (default: False)
- `sentinel`: Sentinel value for unreachable nodes (default: -9999). Must be a negative integer that fits in int32 range.

**Note**: Weight column is not used by BFS. If present in the DataFrame, it will be ignored.

#### Methods

##### run

```python
def run(self, vertex_idx, path_tracking=False, return_series=False)
```

Runs BFS from/to the specified vertex.

**Parameters:**
- `vertex_idx`: Source/target vertex index
- `path_tracking`: Whether to track paths for reconstruction (default: False)
- `return_series`: Return results as pandas Series (default: False)

**Returns:**
- Array or Series of predecessor indices
  - The sentinel value (default `-9999`) indicates unreachable vertex or start vertex
  - Other values indicate the predecessor in the shortest path

##### get_path

```python
def get_path(self, vertex_idx)
```

Reconstructs the shortest path to/from a vertex (requires `path_tracking=True`).

**Parameters:**
- `vertex_idx`: Destination/source vertex index

**Returns:**
- Array of vertex indices forming the path (from target to source)

##### get_vertices

```python
def get_vertices(self)
```

Returns all vertices in the directed graph.

**Returns:**
- Array of vertex indices

#### Examples

##### Basic BFS

```python
from edsger.path import BFS
import pandas as pd

# Create a directed graph
edges = pd.DataFrame({
    'tail': [0, 0, 1, 2, 2, 3],
    'head': [1, 2, 3, 3, 4, 4]
})

# Initialize BFS
bfs = BFS(edges)

# Find shortest paths (by hop count) from vertex 0
predecessors = bfs.run(vertex_idx=0)
print(predecessors)  # [-9999     0     0     1     2]
```

##### Custom Sentinel Value

```python
# Create BFS with custom sentinel for unreachable nodes
bfs = BFS(edges, sentinel=-1)
predecessors = bfs.run(vertex_idx=0)

# Check the sentinel value
print(bfs.UNREACHABLE)  # -1
# Unreachable nodes will have value -1 instead of -9999
```

##### With Path Tracking

```python
# Enable path tracking
predecessors = bfs.run(vertex_idx=0, path_tracking=True)

# Get the actual path to vertex 4
path = bfs.get_path(vertex_idx=4)
print(path)  # [4 2 0] - path from target to source
```

##### With Non-Contiguous Vertex IDs

```python
# Directed graph with non-contiguous vertex IDs
edges_sparse = pd.DataFrame({
    'tail': [0, 0, 10, 20],
    'head': [10, 20, 20, 30]
})

# Use permute=True for efficiency
bfs = BFS(edges_sparse, permute=True)
predecessors = bfs.run(vertex_idx=0, path_tracking=True)
path = bfs.get_path(vertex_idx=30)
print(path)  # [30 20  0]
```

##### Backward Search (orientation='in')

```python
# Find vertices that can reach a target
bfs_in = BFS(edges, orientation='in')
predecessors = bfs_in.run(vertex_idx=4)
# Shows successors in backward traversal
```

##### Return as Pandas Series

```python
# Get results as labeled Series
predecessors = bfs.run(vertex_idx=0, return_series=True)
print(predecessors)
# vertex_idx
# 0       -9999
# 1           0
# 2           0
# 3           1
# 4           2
# Name: predecessor, dtype: int32
```

#### Performance Characteristics

- **Time Complexity**: O(V + E) - linear in vertices and edges
- **Space Complexity**: O(V) - queue and predecessor arrays

### HyperpathGenerating Class

The class for computing hyperpath-based routing using the Spiess-Florian algorithm, designed for transit network assignment and optimal strategy computation.

**Reference**: Spiess, H. and Florian, M. (1989). *Optimal strategies: A new assignment model for transit networks*. Transportation Research Part B 23(2), 83-102.

#### Overview

The Spiess-Florian algorithm computes optimal travel strategies in transit networks where passengers can choose between multiple routes with different frequencies and travel times. Unlike traditional shortest path algorithms, it models the probabilistic nature of transit where passengers may take the first arriving vehicle among a set of attractive lines.

The algorithm outputs:
- **Edge volumes**: How passenger demand is distributed across transit lines
- **Travel times (u_i_vec)**: Expected travel time from each vertex to the destination, including waiting time

#### Constructor Parameters

- `edges`: pandas.DataFrame containing transit network edges
- `tail`: Column name for edge source nodes (default: 'tail')
- `head`: Column name for edge destination nodes (default: 'head')
- `trav_time`: Column name for travel times (default: 'trav_time')
- `freq`: Column name for service frequencies (default: 'freq')
- `check_edges`: Whether to validate edge data (default: False)
- `orientation`: Currently only 'in' is supported (default: 'in')

#### Attributes

- `edge_count`: Number of edges in the graph
- `vertex_count`: Number of vertices in the graph
- `u_i_vec`: Array of expected travel times from each vertex to the destination (available after `run()`)
- `_edges`: DataFrame containing edges with computed 'volume' column (available after `run()`)

#### Methods

##### run

```python
def run(self, origin, destination, volume, return_inf=False)
```

Computes the hyperpath and distributes demand volume across edges.

**Parameters:**
- `origin`: Origin vertex index (int) or list of origin indices
- `destination`: Destination vertex index (int)
- `volume`: Demand volume (float) or list of volumes corresponding to each origin
- `return_inf`: Whether to keep infinity for unreachable vertices (default: False)

**Returns:**
- None (results are stored in instance attributes `u_i_vec` and `_edges['volume']`)

#### Examples

##### Basic Transit Network

```python
from edsger.path import HyperpathGenerating
import pandas as pd

# Create a simple transit network
# Two parallel lines from stop 0 to stop 2 via different intermediate stops
edges = pd.DataFrame({
    'tail': [0, 0, 1, 3],
    'head': [1, 3, 2, 2],
    'trav_time': [5.0, 4.0, 3.0, 4.0],  # Travel times in minutes
    'freq': [0.1, 0.15, 0.2, 0.2]  # Frequencies (vehicles per minute)
})

# Initialize the algorithm
hp = HyperpathGenerating(edges)

# Compute hyperpath from origin 0 to destination 2 with 100 passengers
hp.run(origin=0, destination=2, volume=100.0)

# Access expected travel times to destination
print("Travel times to destination:", hp.u_i_vec)

# Access edge volumes (how passengers are distributed)
print(hp._edges[['tail', 'head', 'volume']])
```

##### Multiple Origins

```python
# Compute hyperpath from multiple origins to a single destination
hp = HyperpathGenerating(edges)
hp.run(
    origin=[0, 1],           # Two origins
    destination=2,           # Single destination
    volume=[80.0, 20.0]      # Different demand volumes
)

# Total edge volumes reflect combined demand from all origins
print("Edge volumes:", hp._edges['volume'].values)
```

##### Custom Column Names

```python
# Transit data with custom column names
transit_edges = pd.DataFrame({
    'from_stop': [0, 0, 1, 2],
    'to_stop': [1, 2, 2, 3],
    'travel_time': [2.0, 5.0, 1.5, 3.0],
    'frequency': [0.1, 0.05, 0.2, 0.15]
})

hp = HyperpathGenerating(
    transit_edges,
    tail='from_stop',
    head='to_stop',
    trav_time='travel_time',
    freq='frequency'
)
hp.run(origin=0, destination=3, volume=1.0)
```

##### With Edge Validation

```python
# Enable input validation
hp = HyperpathGenerating(edges, check_edges=True)

# This will raise an error if:
# - Required columns are missing
# - Data types are incorrect
# - Travel times or frequencies are negative
# - Missing values exist
```

#### Understanding the Output

After calling `run()`, two key outputs are available:

1. **`u_i_vec`**: Expected travel time from each vertex to the destination
   - Includes both in-vehicle travel time and expected waiting time
   - Infinite values indicate unreachable vertices

2. **`_edges['volume']`**: Passenger volumes on each edge
   - Shows how total demand is distributed across the transit network
   - Higher volumes indicate more heavily used transit lines

#### Use Cases

- **Transit assignment**: Distribute passenger demand across a transit network
- **Line planning**: Identify heavily used routes that may need more capacity
- **Service evaluation**: Assess expected travel times for different origin-destination pairs
- **Network design**: Evaluate the impact of adding or removing transit lines

## Parallel Edges Handling

The `Dijkstra`, `BellmanFord`, and `BFS` classes automatically handle parallel edges (multiple directed edges between the same pair of vertices) during initialization. When parallel edges are detected:

1. **Automatic Processing**: The `_preprocess_edges()` method is called internally during initialization
2. **Minimum Weight Selection**: For each pair of vertices with multiple edges, only the edge with the minimum weight is kept
3. **Verbose Output**: If `verbose=True` is set in the constructor, a message will be printed indicating how many parallel edges were removed

### Example with Parallel Edges

```python
import pandas as pd
from edsger.path import Dijkstra

# Directed graph with parallel edges between vertices 0->1
edges = pd.DataFrame({
    'tail': [0, 0, 1],     # Two directed edges from 0->1
    'head': [1, 1, 2],     # with weights 5.0 and 3.0
    'weight': [5.0, 3.0, 2.0]
})

# Initialize with verbose=True to see parallel edge removal
dijkstra = Dijkstra(edges, verbose=True)
# Output: "Automatically removed 1 parallel edge(s). For each pair of vertices, kept the edge with minimum weight."

# The directed graph now has only the edge 0->1 with weight 3.0 (minimum)
print(dijkstra.n_edges)  # Will show 2 instead of 3
```

This automatic handling ensures:
- Consistent directed graph representation
- Optimal shortest paths (using minimum weight edges)
- No duplicate edges in the internal directed graph structure

## edsger.graph_importer Module

### GraphImporter Architecture

The GraphImporter system provides automatic DataFrame backend detection and optimization for all graph algorithms in Edsger.

#### Key Features

- **Automatic Detection**: Identifies DataFrame type (pandas NumPy, pandas Arrow, or Polars)
- **Memory Optimization**: Automatically uses uint32 for vertex indices when possible
- **Contiguous Memory**: Ensures C-contiguous arrays for optimal Cython performance
- **Factory Pattern**: Simple interface with automatic backend selection

### standardize_graph_dataframe Function

```python
from edsger.graph_importer import standardize_graph_dataframe

def standardize_graph_dataframe(edges, tail='tail', head='head', weight='weight')
```

Standardizes any supported DataFrame format to NumPy-backed pandas DataFrame.

**Parameters:**
- `edges`: Input DataFrame (pandas, pandas with Arrow, or Polars)
- `tail`: Column name for source vertices
- `head`: Column name for destination vertices
- `weight`: Column name for edge weights

**Returns:**
- pandas DataFrame with NumPy arrays, optimized dtypes, and contiguous memory

#### Example Usage

```python
import pandas as pd
import polars as pl
from edsger.graph_importer import standardize_graph_dataframe

# Works with any DataFrame type
edges_polars = pl.DataFrame({
    'source': [0, 1, 2],
    'target': [1, 2, 3],
    'cost': [1.0, 2.0, 3.0]
})

# Automatic conversion and optimization
edges_numpy = standardize_graph_dataframe(
    edges_polars,
    tail='source',
    head='target',
    weight='cost'
)

print(edges_numpy.dtypes)
# source    uint32  (optimized from int64)
# target    uint32  (optimized from int64)
# cost      float64
```

### GraphImporter Classes

#### GraphImporter (Base Class)

Abstract base class providing the factory method for automatic DataFrame type detection.

##### Class Method: from_dataframe

```python
@staticmethod
def from_dataframe(edges, tail='tail', head='head', weight='weight')
```

Factory method that automatically selects the appropriate importer based on DataFrame type.

**Detection Priority:**
1. Polars DataFrame
2. pandas with Arrow backend
3. pandas with NumPy backend (default)

#### PandasNumpyImporter

Handles standard pandas DataFrames with NumPy backend.

- Minimal overhead for already NumPy-backed DataFrames
- Ensures contiguous memory layout
- Maintains existing dtypes

#### PandasArrowImporter

Handles pandas DataFrames with PyArrow backend.

- Converts Arrow columns to NumPy arrays
- Optimizes integer dtypes (uint32 when possible)
- Handles nullable types gracefully

#### PolarsImporter

Handles Polars DataFrames.

- Efficient conversion using Polars' to_pandas() method
- Automatic dtype optimization
- Preserves performance benefits of columnar storage during conversion

### Performance Considerations

The GraphImporter system optimizes for:

1. **Memory Efficiency**: Uses smallest possible dtype for vertex indices
2. **Cache Locality**: Ensures C-contiguous memory layout
3. **Conversion Speed**: Single conversion at initialization
4. **Zero-Copy When Possible**: Minimizes data copying during conversion

### Integration with Algorithms

All path algorithms (Dijkstra, BellmanFord, BFS, HyperpathGenerating) automatically use the GraphImporter system:

```python
from edsger.path import Dijkstra
import polars as pl

# GraphImporter is used internally
edges = pl.DataFrame({...})
dijkstra = Dijkstra(edges)  # Automatic detection and conversion
```

No manual conversion is needed - the algorithms handle everything automatically!
