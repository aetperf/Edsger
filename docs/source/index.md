---
github_url: https://github.com/aetperf/Edsger
---

# Edsger

*Directed graph algorithms in Cython*

Welcome to the Edsger documentation! Edsger is a Python library for efficient graph algorithms implemented in Cython. The library focuses on shortest path algorithms, featuring so far both Dijkstra's algorithm for positive-weight directed graphs and Bellman-Ford algorithm for directed graphs with negative weights and cycle detection.

## Why Use Edsger?

Edsger is designed to be **dataframe-friendly**, providing seamless integration with pandas workflows for directed graph algorithms. Also it is rather efficient on Linux. Our benchmarks on the USA road network (23.9M vertices, 57.7M edges) demonstrate nice performance:

<img src="assets/dijkstra_benchmark_comparison.png" alt="Dijkstra Performance Comparison" width="700">

### Pandas Integration Made Simple

```python
import pandas as pd
from edsger.path import Dijkstra, BellmanFord

# Your graph data is already in a DataFrame
edges = pd.DataFrame({
    'tail': [0, 0, 1, 2],
    'head': [1, 2, 2, 3], 
    'weight': [1.0, 2.0, 1.5, 1.0]
})

# Use Dijkstra for positive weights (faster)
dijkstra = Dijkstra(edges)
distances = dijkstra.run(vertex_idx=0)

# Use Bellman-Ford for negative weights or cycle detection
bf = BellmanFord(edges)
distances = bf.run(vertex_idx=0)
```

## Key Features
s
- **Native pandas DataFrame support** - No graph object conversion required
- **High performance** - Cython implementation
- **Memory efficient** - Optimized for real-world datasets
- **Easy integration** with NumPy and Pandas workflows
- **Production ready** - Comprehensive testing across Python 3.9-3.13

## Quick Links

- [Installation](installation.md) - How to install Edsger
- [Quick Start](quickstart.md) - Get started quickly with basic examples
- [API Reference](api.md) - Complete API reference

## Table of Contents

```{toctree}
:maxdepth: 2
:caption: User Guide

installation
quickstart
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api
```

```{toctree}
:maxdepth: 1
:caption: Development

contributing
```

## Indices

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`

