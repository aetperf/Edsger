# Benchmarks

Dijkstra's algorithm is timed on the USA road network of the [9th DIMACS implementation challenge](https://archive.dimacs.rutgers.edu/Workshops/Challenge9/), with the travel time as edge weight. This graph has 23,947,347 vertices and 57,708,624 edges. Edsger is compared with [NetworKit](https://networkit.github.io/), [graph-tool](https://graph-tool.skewed.de/) and [SciPy](https://scipy.org/).

## Method

Each library computes the shortest path lengths from vertex 1000 to all the vertices, without storing the paths. Only this computation is timed: loading the network and building the graph of each library are left out. The reported time is the smallest of 5 runs.

| Library | Call |
|---|---|
| Edsger | `Dijkstra(edges).run(vertex_idx=1000)` |
| NetworKit | `nk.distance.Dijkstra(g, 1000, storePaths=False, storeNodesSortedByDistance=False).run()` |
| graph-tool | `shortest_distance(g, source=g.vertex(1000), weights=g.ep.t, negative_weights=False, directed=True)` |
| SciPy | `scipy.sparse.csgraph.dijkstra(csgraph=graph_csr, directed=True, indices=1000, return_predecessors=False)` |

## Results

```{image} assets/dijkstra_benchmark_comparison.png
:alt: Run time of Dijkstra's algorithm on the USA road network, for Edsger, NetworKit, graph-tool and SciPy
:width: 700px
```

In the figure, a bar is the best of the 5 runs and its whisker goes up to the slowest one. The table gives the best run times in seconds, measured on 2026-11-12.

| Library | Version | Linux | Windows |
|---|---|---:|---:|
| Edsger | 0.2.0 (development) | 1.05 | 1.06 |
| SciPy | 1.18.1 | 2.86 | 2.81 |
| graph-tool | 3.9 | 4.89 | |
| NetworKit | 11.3 | 5.08 | 5.13 |

graph-tool is not available on Windows.

The Linux and Windows columns come from two different machines, so they do not compare the operating systems. The libraries can be compared within a column.

| | Linux | Windows |
|---|---|---|
| CPU | Intel Core i9-12900H (laptop) | Intel Core i7-14700K (desktop) |
| System | Linux 5.15, glibc 2.35 | Windows 11 |
| Python | 3.14.2 | 3.14.10 |
| NumPy, pandas | 2.5.4, 3.1.1 | 2.5.4, 3.1.1 |
| Edsger build | From source, Clang 14, `-march=native` | From source |

## Compilers

On Linux, the compiler has a visible effect on the Dijkstra code. The following timings were taken on the same i9-12900H with the same source code and the same graph, the process being pinned to one core, best of 7 runs.

| Build | Run time (s) |
|---|---:|
| GCC 11, `-march=native` | 1.22 |
| Clang 14, `-march=native` | 1.07 |
| GCC 14, baseline x86-64 | 1.18 |
| Clang 21, baseline x86-64 | 1.08 |

The Linux wheels on PyPI are built as in the last row. The path lengths are bit-identical in the four cases.

## Running the benchmark

The scripts are in the [`scripts`](https://github.com/aetperf/Edsger/tree/release/scripts) directory of the repository, with the pinned versions of the packages in `requirements_linux.txt` and `requirements_windows.txt`.

```bash
cd scripts
python benchmark_comparison_os.py -d /path/to/DIMACS_road_networks -n USA -r 5
python plot_benchmark_comparison.py
```

The first script reads the network from the Parquet file `USA/USA-road-t.USA.gr.parquet` of the data directory, runs the four libraries and writes the timings to a JSON file. The second one draws the figure from the JSON files that it finds.

The download of the DIMACS networks and their conversion to Parquet files are described in a [blog post](https://architecture-performance.fr/en/blog/download-some-benchmark-road-networks-for-shortest-paths-algorithms).
