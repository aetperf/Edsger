# Transit assignment

This example reproduces the network of [Spiess and Florian (1989)](https://doi.org/10.1016/0191-2615(89)90034-9), the paper that introduced optimal strategies for frequency-based transit networks, and assigns a demand to it with `HyperpathGenerating`.

## The network

Four stops, A, X, Y and B, are served by four lines:

| Line | Route | Ride times (minutes) | Headway (minutes) |
|---|---|---|---|
| 1 | A - B | 25 | 12 |
| 2 | A - X - Y | 7, 6 | 12 |
| 3 | X - Y - B | 4, 4 | 30 |
| 4 | Y - B | 10 | 6 |

In the graph, each stop has a stop vertex, and each line at each stop has a boarding or an alighting vertex. Boarding edges carry the frequency of the line; the other edges (riding, alighting, dwelling, transferring) have an infinite frequency, so no waiting time. `create_sf_network` builds this graph, with times in seconds, and adds the edge volumes given in the paper for a demand of 1 from A (vertex 0) to B (vertex 12).

```python
from edsger.networks import create_sf_network

edges = create_sf_network()
edges.shape
```

% output

    (24, 5)

% /output

## Assignment

`run` computes the optimal strategy to the destination, and assigns the demand of each origin to the edges.

```python
from edsger import HyperpathGenerating

hp = HyperpathGenerating(edges)
hp.run(origin=0, destination=12, volume=1.0)
```

The expected travel time from each vertex to the destination, waiting included, is stored in `u_i_vec`. From the stop vertices of A, X and Y, in minutes:

```python
stops = {"A": 0, "X": 3, "Y": 7}
{stop: round(float(hp.u_i_vec[v]) / 60.0, 2) for stop, v in stops.items()}
```

% output

    {'A': 27.75, 'X': 19.07, 'Y': 11.5}

% /output

These are the values of the paper: 27.75 minutes from A to B.

## Edge volumes

The volumes are stored in the `volume` column of `edges`. The edges used by the strategy:

```python
volumes = hp.edges.assign(volume_ref=edges["volume_ref"])
used = volumes[volumes["volume"] > 1e-12]
used[["tail", "head", "trav_time", "volume", "volume_ref"]].round(4)
```

% output

        tail  head  trav_time  volume  volume_ref
    0      0     2        0.0  0.5000      0.5000
    1      0     1        0.0  0.5000      0.5000
    2      2    15     1500.0  0.5000      0.5000
    3      1     5      420.0  0.5000      0.5000
    5      5     6        0.0  0.5000      0.5000
    9      6    11      360.0  0.5000      0.5000
    13    11    10        0.0  0.0833      0.0833
    14    11     8        0.0  0.4167      0.4167
    19    10    14      240.0  0.0833      0.0833
    20     8    13      600.0  0.4167      0.4167
    21    15    12        0.0  0.5000      0.5000
    22    14    12        0.0  0.0833      0.0833
    23    13    12        0.0  0.4167      0.4167

% /output

At stop A, line 1 and line 2 are both attractive and have the same frequency, so the demand is split in half. The travellers on line 2 ride to Y, where lines 3 and 4 are attractive: they board the first vehicle to come, so 1/12 take line 3 (every 30 minutes) and 5/12 line 4 (every 6 minutes), in proportion to the frequencies. All the volumes match the paper:

```python
import numpy as np

np.allclose(volumes["volume"], volumes["volume_ref"])
```

% output

    True

% /output

## Several origins

The demand of several origins to the same destination is assigned in one run, with one volume per origin. The volumes add up on the shared edges.

```python
hp.run(origin=[0, 3], destination=12, volume=[1.0, 2.0])
hp.edges.loc[hp.edges["volume"] > 1e-12, ["tail", "head", "volume"]].round(4)
```

% output

        tail  head  volume
    0      0     2  0.5000
    1      0     1  0.5000
    2      2    15  0.5000
    3      1     5  0.5000
    5      5     6  0.5000
    7      3     6  1.4286
    8      3     4  0.5714
    9      6    11  1.9286
    10     4     9  0.5714
    13    11    10  0.3214
    14    11     8  1.6071
    16     9    10  0.5714
    19    10    14  0.8929
    20     8    13  1.6071
    21    15    12  0.5000
    22    14    12  0.8929
    23    13    12  1.6071

% /output
