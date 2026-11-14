# Origin-destination matrix

Transport models need the travel time between every pair of zones, often called a skim matrix. This example computes it on the Sioux Falls road network with `Dijkstra.run_many`, then combines it with a demand table.

## The network

Sioux Falls is a classical test network of the traffic assignment literature, shipped with Edsger: 24 vertices, which are also the zones, and 76 edges with a travel time in minutes.

```python
from edsger.networks import SiouxFalls

edges = SiouxFalls().edges
edges.head()
```

% output

       tail  head  trav_time
    0     0     1   6.000816
    1     0     2   4.008691
    2     1     0   6.000834
    3     1     5   6.573598
    4     2     0   4.008587

% /output

## The travel time matrix

`run_many` runs Dijkstra's algorithm from each origin, in parallel threads. With `output="pandas"`, the result is a DataFrame with one row per origin and one column per destination.

```python
from edsger import Dijkstra

dijkstra = Dijkstra(edges, weight="trav_time")
zones = dijkstra.get_vertices()
skim = dijkstra.run_many(zones, output="pandas")
skim.iloc[:5, :5].round(2)
```

% output

    destination      0      1      2      3      4
    origin
    0             0.00   6.00   4.01   8.28  10.59
    1             6.00   0.00  10.01  14.28  16.59
    2             4.01  10.01   0.00   4.27   6.58
    3             8.28  14.28   4.27   0.00   2.32
    4            10.60  16.60   6.59   2.32   0.00

% /output

When only some destinations are needed, `termination_nodes` stops each run as soon as they are reached, and the matrix has one column per destination:

```python
dijkstra.run_many([0, 1, 2], termination_nodes=[19, 23], output="pandas").round(2)
```

% output

    destination     19     23
    origin
    0            39.09  28.71
    1            33.09  34.71
    2            43.10  24.70

% /output

## Joining with a demand table

With `output="polars"`, the matrix is returned in long format, one row per origin-destination pair, which joins directly with a table of trips. The demand below is synthetic.

```python
import numpy as np
import polars as pl

rng = np.random.default_rng(0)
origins, destinations = np.meshgrid(zones, zones, indexing="ij")
demand = pl.DataFrame(
    {
        "origin": origins.ravel(),
        "destination": destinations.ravel(),
        "trips": rng.poisson(50, size=origins.size),
    }
).filter(pl.col("origin") != pl.col("destination"))

times = dijkstra.run_many(zones, output="polars")
trips = demand.join(times, on=["origin", "destination"])
trips.head()
```

% output

    shape: (5, 4)
    ┌────────┬─────────────┬───────┬─────────────┐
    │ origin ┆ destination ┆ trips ┆ path_length │
    │ ---    ┆ ---         ┆ ---   ┆ ---         │
    │ u32    ┆ u32         ┆ i64   ┆ f64         │
    ╞════════╪═════════════╪═══════╪═════════════╡
    │ 0      ┆ 1           ┆ 32    ┆ 6.000816    │
    │ 0      ┆ 2           ┆ 52    ┆ 4.008691    │
    │ 0      ┆ 3           ┆ 51    ┆ 8.278093    │
    │ 0      ┆ 4           ┆ 57    ┆ 10.593467   │
    │ 0      ┆ 5           ┆ 59    ┆ 12.574414   │
    └────────┴─────────────┴───────┴─────────────┘

% /output

The total travel time, in vehicle-hours, and the average trip duration in minutes:

```python
minutes = pl.col("trips") * pl.col("path_length")
trips.select(
    veh_hours=(minutes.sum() / 60).round(1),
    mean_minutes=(minutes.sum() / pl.col("trips").sum()).round(2),
)
```

% output

    shape: (1, 2)
    ┌───────────┬──────────────┐
    │ veh_hours ┆ mean_minutes │
    │ ---       ┆ ---          │
    │ f64       ┆ f64          │
    ╞═══════════╪══════════════╡
    │ 11471.9   ┆ 24.73        │
    └───────────┴──────────────┘

% /output

## Large networks

- The matrix is dense: 8 bytes per pair of vertices. Without `termination_nodes`, a NumPy result has one column per integer between 0 and the largest vertex identifier; use `permute=True` when the identifiers are sparse.
- `n_jobs` sets the number of threads, all the CPUs by default. The runs are independent and release the GIL; the speed-up is limited by the memory bandwidth on large graphs (about 3.5 times with 8 threads on a random graph with 1 million vertices).
- For the travel times to a set of destinations from all the vertices, build the graph with `orientation="in"` and run from the destinations: each run then gives one column of the matrix.
