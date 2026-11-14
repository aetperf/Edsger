# Routing on a road network

This example computes travel times and routes on a road network: from one vertex to all the others with `Dijkstra`, and between two vertices with `BidirectionalDijkstra` and `AStar`.

## The network

The network is a synthetic grid of 200 x 200 intersections, 100 meters apart, with a two-way road between neighbouring intersections. Each road has a speed limit of 30, 50 or 90 km/h, and the edge weight is the travel time in seconds. Each vertex has coordinates in meters.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
size, spacing = 200, 100.0

idx = np.arange(size * size).reshape(size, size)
coordinates = pd.DataFrame(
    {
        "x": np.tile(np.arange(size), size) * spacing,
        "y": np.repeat(np.arange(size), size) * spacing,
    }
)

# horizontal and vertical roads, in both directions
pairs = np.vstack(
    [
        np.column_stack((idx[:, :-1].ravel(), idx[:, 1:].ravel())),
        np.column_stack((idx[:-1, :].ravel(), idx[1:, :].ravel())),
    ]
)
speed = rng.choice([30.0, 50.0, 90.0], size=len(pairs), p=[0.5, 0.4, 0.1])
travel_time = spacing / (speed / 3.6)  # seconds
edges = pd.DataFrame(
    {
        "tail": np.concatenate((pairs[:, 0], pairs[:, 1])),
        "head": np.concatenate((pairs[:, 1], pairs[:, 0])),
        "weight": np.concatenate((travel_time, travel_time)),
    }
)
edges.shape
```

% output

    (159200, 3)

% /output

## Travel times from one vertex

`Dijkstra` computes the travel time from a vertex to every vertex of the network, here from the south-west corner.

```python
from edsger import Dijkstra

dijkstra = Dijkstra(edges)
travel_times = dijkstra.run(vertex_idx=0)
round(float(travel_times.max()) / 60.0, 1)  # minutes to the farthest intersection
```

% output

    43.1

% /output

With `orientation="in"`, the same class gives the travel times from every vertex to a destination, for instance to find the area within 10 minutes of a facility:

```python
facility = idx[100, 100]
to_facility = Dijkstra(edges, orientation="in").run(vertex_idx=facility)
int((to_facility <= 600.0).sum())  # intersections within 10 minutes
```

% output

    14087

% /output

## Route between two vertices

When only one route is needed, `BidirectionalDijkstra` searches from both ends and stops when the two searches meet. Its working arrays are kept between runs, so a query only costs time for the part of the network it explores.

```python
from edsger import BidirectionalDijkstra

origin, destination = idx[20, 30], idx[150, 170]

bidir = BidirectionalDijkstra(edges)
round(bidir.run(source=origin, target=destination, path_tracking=True), 3)
```

% output

    1761.6

% /output

The route is a list of vertices, from the origin to the destination. Joined with the coordinates, it gives the geometry of the route:

```python
route = bidir.get_path()
route_xy = coordinates.loc[route]
len(route), route_xy.iloc[[0, -1]]
```

% output

    (271,              x        y
    4030    3000.0   2000.0
    30170  17000.0  15000.0)

% /output

## A* with coordinates

`AStar` uses the coordinates to search towards the destination first. `set_coordinates` derives the heuristic from them: the straight-line distance to the destination times a factor, computed from the edges so that the heuristic never overestimates the travel time. Here the factor is the inverse of the largest speed, 90 km/h, in seconds per meter.

```python
from edsger import AStar

astar = AStar(edges)
factor = astar.set_coordinates(coordinates, x="x", y="y")
round(3.6 / factor, 3)  # the largest speed, in km/h
```

% output

    90.0

% /output

```python
round(astar.run(source=origin, target=destination, path_tracking=True), 3)
```

% output

    1761.6

% /output

The two algorithms find the same travel time, about 29 minutes, and so does `Dijkstra` with a termination node:

```python
dijkstra.run(vertex_idx=origin, termination_nodes=[destination])
```

% output

    array([1761.6])

% /output

For coordinates given as longitudes and latitudes in degrees, `metric="haversine"` uses the great-circle distance in meters.

## Which algorithm?

- From one vertex to many others: `Dijkstra.run`, or `run_many` for several sources in parallel.
- Between two vertices far apart on a large network: `Dijkstra` with `termination_nodes`, `BidirectionalDijkstra` or `AStar` take similar times; the gain of A* depends on how tight the heuristic is. On a travel-time network, the straight-line distance divided by the largest speed is a loose bound.
- Many queries between nearby vertices: `BidirectionalDijkstra` and `AStar`, whose cost does not depend on the size of the network.
