# Transit assignment

The `HyperpathGenerating` class implements the algorithm of [Spiess and Florian (1989)](https://doi.org/10.1016/0191-2615(89)90034-9) for frequency-based transit networks. Each edge has a travel time and a frequency. At a stop, a traveller boards the first vehicle to come among a set of attractive lines, so the travel time includes an expected waiting time, and the demand is split between the lines in proportion to their frequencies.

| Column | Type | Content |
|---|---|---|
| `tail` | integer | Start vertex of the edge |
| `head` | integer | End vertex of the edge |
| `trav_time` | float | Travel time on the edge |
| `freq` | float | Frequency of the edge, in vehicles per unit of travel time |

In the following network, times are in minutes. A direct line goes from vertex 0 to vertex 2 in 25 minutes, every 12 minutes. Another line goes from 0 to 1 in 7 minutes, every 12 minutes, and a third one from 1 to 2 in 10 minutes, every 6 minutes.

```python
import pandas as pd

from edsger import HyperpathGenerating

transit = pd.DataFrame(
    {
        "tail": [0, 0, 1],
        "head": [2, 1, 2],
        "trav_time": [25.0, 7.0, 10.0],
        "freq": [1 / 12, 1 / 12, 1 / 6],
    }
)
hp = HyperpathGenerating(transit)
hp.run(origin=0, destination=2, volume=100.0)
```

The `run` method returns nothing. The expected travel time from each vertex to the destination is stored in `u_i_vec`.

```python
hp.u_i_vec
```

% output

    array([30., 16.,  0.])

% /output

From vertex 1, the expected waiting time is 6 minutes and the ride takes 10 minutes. From vertex 0, both lines are attractive: the first vehicle comes after 6 minutes on average, and half of the travellers end up on each line, with 25 and 23 minutes left to the destination. The edge volumes are stored in the `volume` column of the `edges` dataframe.

```python
hp.edges[["tail", "head", "volume"]]
```

% output

       tail  head  volume
    0     0     2    50.0
    1     0     1    50.0
    2     1     2    50.0

% /output

Several origins can be assigned in one run, with one volume per origin.

```python
hp.run(origin=[0, 1], destination=2, volume=[100.0, 50.0])
hp.edges[["tail", "head", "volume"]]
```

% output

       tail  head  volume
    0     0     2    50.0
    1     0     1    50.0
    2     1     2   100.0

% /output


For a complete network with stops, lines and transfers, see the [transit assignment example](../examples/transit_assignment.md).
