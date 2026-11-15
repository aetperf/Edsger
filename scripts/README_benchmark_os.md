# OS-Specific Benchmarking Scripts

These scripts allow you to run benchmarks on different operating systems and combine the results into comparison plots.

## Scripts

1. **benchmark_comparison_os.py** - Runs benchmarks and creates OS-specific JSON files
2. **plot_benchmark_comparison.py** - Creates comparison plots from the JSON files

## Usage

### Step 1: Run benchmarks on each OS

On Linux:
```bash
cd scripts/
python benchmark_comparison_os.py [-d DATA_DIR] [-n NETWORK] [-r REPEAT]
```
This creates: `benchmark_dimacs_USA_linux.json` (or with different network name)

On Windows:
```bash
cd scripts/
python benchmark_comparison_os.py [-d DATA_DIR] [-n NETWORK] [-r REPEAT]
```
This creates: `benchmark_dimacs_USA_windows.json` (or with different network name)

On macOS:
```bash
cd scripts/
python benchmark_comparison_os.py [-d DATA_DIR] [-n NETWORK] [-r REPEAT]
```
This creates: `benchmark_dimacs_USA_darwin.json` (or with different network name)

#### Arguments:
- `-d, --dir`: Data folder with network sub-folders (default: from DIMACS_DATA_DIR env var or `/home/francois/Data/DIMACS_road_networks/`)
- `-n, --network`: Network name - 'NY', 'BAY', 'COL', 'FLA', 'NW', 'NE', 'CAL', 'LKS', 'E', 'W', 'CTR', 'USA' (default: USA)
- `-r, --repeat`: Number of benchmark iterations per library (default: 5)

### Step 2: Create comparison plots

After collecting results from one or more operating systems:
```bash
python plot_benchmark_comparison.py
```

This will create:
- Combined comparison plot: `dijkstra_benchmark_comparison.png`

## Memory

Each library run also measures its memory (`memory_usage.py`), as the increase of the resident set size (RSS) of the process over the loaded edge list:

- **graph**: the memory kept once the graph is built (`retained_mb`), and the peak during the build (`peak_mb`);
- **run**: the peak during a Dijkstra run, the distance array included (largest of the runs).

On Linux, the peak is exact: the peak RSS (`VmHWM`) is reset before each phase by writing to `/proc/self/clear_refs`. Elsewhere it is sampled every millisecond with psutil (`pip install psutil`). With glibc, the memory freed but kept by the allocator is returned to the system before each phase (`malloc_trim`), so that a phase cannot reuse it unseen. The values go to the `memory` field of each library in the JSON file, and `plot_benchmark_comparison.py` draws them in `dijkstra_memory_comparison.png`. For Edsger, `graph_data_mb` gives the size of its arrays: the edges dataframe it keeps and the compressed graph.

When Edsger is measured from a development build, `--edsger-release X.Y.Z` records the release with the same code, shown in the figures instead of the development version.

## Features

- **Graceful handling of missing packages**: Graph-tool is automatically skipped on Windows
- **OS detection**: Automatically names output files based on the operating system
- **Multi-OS comparison**: Can compare results across Linux, Windows, and macOS
- **Version tracking**: Records package versions for each benchmark run

## Output Files

### JSON files
- `benchmark_dimacs_{NETWORK}_{OS}.json` - Benchmark results for specific network and OS
  - Examples: `benchmark_dimacs_USA_linux.json`, `benchmark_dimacs_COL_windows.json`

### Plot files
- `dijkstra_benchmark_comparison.png` - Comparison plot (shows all available OS results)

## Notes

- **Graph-tool**: Not available on Windows and will be automatically skipped
- **SciPy benchmarking**: Runs `dijkstra_dimacs.py` with Edsger + comparison (`-c` flag) multiple times to collect SciPy timing statistics, since SciPy only runs once per call in comparison mode
- **Package detection**: Scripts detect available packages before running benchmarks
- **Metadata**: Results include system information (with the processor model, shown in the plot legend), Python version, and package versions
- **Plot**: Each bar is the best run and its whisker goes up to the slowest run. The date on the plot is the date of the benchmark runs, read from the JSON files
- **Statistical reliability**: Benchmarks run 5 iterations by default for each library