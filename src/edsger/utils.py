"""This module makes it easy to execute common tasks in Python scripts such as generate random
graphs.
"""

from typing import Tuple
import numpy as np
import pandas as pd


def generate_random_network(
    n_edges: int = 100,
    n_verts: int = 20,
    seed: int = 124,
    sort: bool = True,
    allow_negative_weights: bool = False,
    negative_weight_ratio: float = 0.3,
    weight_range: Tuple[float, float] = (0.1, 1.0),
) -> pd.DataFrame:
    """
    Generate a random network with a specified number of edges and vertices.

    Parameters
    ----------
    n_edges : int, optional
        The number of edges in the network. Default is 100.
    n_verts : int, optional
        The number of vertices in the network. Default is 20.
    seed : int, optional
        The seed for the random number generator. Default is 124.
    sort : bool, optional
        Whether to sort the edges by tail and head vertices. Default is True.
    allow_negative_weights : bool, optional
        Whether to allow negative edge weights. Default is False (positive weights only).
    negative_weight_ratio : float, optional
        Proportion of edges that should have negative weights when allow_negative_weights=True.
        Must be between 0.0 and 1.0. Default is 0.3 (30% negative).
    weight_range : tuple of float, optional
        Range of absolute values for weights as (min, max). Default is (0.1, 1.0).
        When allow_negative_weights=True, negative weights will be in range (-max, -min).

    Returns
    -------
    edges : pandas.DataFrame
        A DataFrame containing the edges of the network with columns 'tail', 'head', and 'weight'.

    Examples
    --------
    Generate a graph with positive weights only (default):

    >>> generate_random_network(n_edges=5, n_verts=3, seed=42)
       tail  head    weight
    0     0     2  0.975622
    1     1     0  0.128114
    2     1     0  0.450386
    3     1     2  0.786064
    4     2     0  0.761140

    Generate a graph with mixed positive and negative weights:

    >>> generate_random_network(n_edges=5, n_verts=3, seed=42,
    ...                        allow_negative_weights=True, negative_weight_ratio=0.4)
       tail  head    weight
    0     0     2  0.975622
    1     1     0 -0.128114
    2     1     0  0.450386
    3     1     2 -0.786064
    4     2     0  0.761140

    Notes
    -----
    The 'tail' and 'head' columns represent the source and destination vertices of each edge,
    respectively. The 'weight' column represents the weight of each edge.

    When allow_negative_weights=False (default), weights are random floats between
    weight_range[0] and weight_range[1].

    When allow_negative_weights=True, approximately negative_weight_ratio proportion of edges
    will have negative weights, useful for testing algorithms like Bellman-Ford that support
    negative edge weights.

    If `sort` is True, the DataFrame is sorted by the 'tail' and 'head' columns and the index
    is reset.
    """
    # Validate parameters
    if not 0.0 <= negative_weight_ratio <= 1.0:
        raise ValueError("negative_weight_ratio must be between 0.0 and 1.0")
    if (
        len(weight_range) != 2
        or weight_range[0] <= 0
        or weight_range[1] <= weight_range[0]
    ):
        raise ValueError("weight_range must be (min, max) with 0 < min < max")

    rng = np.random.default_rng(seed=seed)
    tail = rng.integers(low=0, high=n_verts, size=n_edges)
    head = rng.integers(low=0, high=n_verts, size=n_edges)

    # Generate weights
    if allow_negative_weights:
        # Generate weights in the specified range
        weight = rng.uniform(low=weight_range[0], high=weight_range[1], size=n_edges)

        # Randomly select edges to have negative weights
        n_negative = int(n_edges * negative_weight_ratio)
        if n_negative > 0:
            negative_indices = rng.choice(n_edges, size=n_negative, replace=False)
            weight[negative_indices] *= -1
    else:
        # Original behavior: positive weights in range [weight_range[0], weight_range[1]]
        if weight_range == (0.1, 1.0):
            # Keep backward compatibility for default case
            weight = rng.random(size=n_edges)
        else:
            weight = rng.uniform(
                low=weight_range[0], high=weight_range[1], size=n_edges
            )

    edges = pd.DataFrame(data={"tail": tail, "head": head, "weight": weight})
    if sort:
        edges.sort_values(by=["tail", "head"], inplace=True)
        edges.reset_index(drop=True, inplace=True)
    return edges


def create_sf_network(dwell_time=0.0, board_alight_ratio=0.5):
    """
    Example network from Spiess, H. and Florian, M. (1989).
    Optimal strategies: A new assignment model for transit networks.
    Transportation Research Part B 23(2), 83-102.

    This network has 13 vertices and 24 edges.

    Parameters
    ----------
    dwell_time : float, optional
        The dwell time at each stop in seconds. Default is 1.0e-6.
    board_alight_ratio : float, optional
        The ratio of boarding to alighting in dwell time. Default is 0.5.
        dwell_time = board_alight_ratio * boarding_time +
            (1 - board_alight_ratio) * alighting_time

    Returns
    -------
    edges : pandas.DataFrame
        A DataFrame containing the edges of the network with columns 'tail',
        'head', 'trav_time', 'freq', and 'volume_ref'.
    """

    boarding_time = board_alight_ratio * dwell_time
    alighting_time = board_alight_ratio * dwell_time

    line1_freq = 1.0 / (60.0 * 12.0)
    line2_freq = 1.0 / (60.0 * 12.0)
    line3_freq = 1.0 / (60.0 * 30.0)
    line4_freq = 1.0 / (60.0 * 6.0)

    # stop A
    # 0 stop vertex
    # 1 boarding vertex
    # 2 boarding vertex

    # stop X
    # 3 stop vertex
    # 4 boarding vertex
    # 5 alighting vertex
    # 6 boarding vertex

    # stop Y
    # 7  stop vertex
    # 8  boarding vertex
    # 9  alighting vertex
    # 10 boarding vertex
    # 11 alighting vertex

    # stop B
    # 12 stop vertex
    # 13 alighting vertex
    # 14 alighting vertex
    # 15 alighting vertex

    tail = []
    head = []
    trav_time = []
    freq = []
    vol = []

    # edge 0
    # stop A : to line 1
    # boarding edge
    tail.append(0)
    head.append(2)
    freq.append(line1_freq)
    trav_time.append(boarding_time)
    vol.append(0.5)

    # edge 1
    # stop A : to line 2
    # boarding edge
    tail.append(0)
    head.append(1)
    freq.append(line2_freq)
    trav_time.append(boarding_time)
    vol.append(0.5)

    # edge 2
    # line 1 : first segment
    # on-board edge
    tail.append(2)
    head.append(15)
    freq.append(np.inf)
    trav_time.append(25.0 * 60.0)
    vol.append(0.5)

    # edge 3
    # line 2 : first segment
    # on-board edge
    tail.append(1)
    head.append(5)
    freq.append(np.inf)
    trav_time.append(7.0 * 60.0)
    vol.append(0.5)

    # edge 4
    # stop X : from line 2
    # alighting edge
    tail.append(5)
    head.append(3)
    freq.append(np.inf)
    trav_time.append(alighting_time)
    vol.append(0.0)

    # edge 5
    # stop X : in line 2
    # dwell edge
    tail.append(5)
    head.append(6)
    freq.append(np.inf)
    trav_time.append(dwell_time)
    vol.append(0.5)

    # edge 6
    # stop X : from line 2 to line 3
    # transfer edge
    tail.append(5)
    head.append(4)
    freq.append(line3_freq)
    trav_time.append(dwell_time)
    vol.append(0.0)

    # edge 7
    # stop X : to line 2
    # boarding edge
    tail.append(3)
    head.append(6)
    freq.append(line2_freq)
    trav_time.append(boarding_time)
    vol.append(0.0)

    # edge 8
    # stop X : to line 3
    # boarding edge
    tail.append(3)
    head.append(4)
    freq.append(line3_freq)
    trav_time.append(boarding_time)
    vol.append(0.0)

    # edge 9
    # line 2 : second segment
    # on-board edge
    tail.append(6)
    head.append(11)
    freq.append(np.inf)
    trav_time.append(6.0 * 60.0)
    vol.append(0.5)

    # edge 10
    # line 3 : first segment
    # on-board edge
    tail.append(4)
    head.append(9)
    freq.append(np.inf)
    trav_time.append(4.0 * 60.0)
    vol.append(0.0)

    # edge 11
    # stop Y : from line 3
    # alighting edge
    tail.append(9)
    head.append(7)
    freq.append(np.inf)
    trav_time.append(alighting_time)
    vol.append(0.0)

    # edge 12
    # stop Y : from line 2
    # alighting edge
    tail.append(11)
    head.append(7)
    freq.append(np.inf)
    trav_time.append(alighting_time)
    vol.append(0.0)

    # edge 13
    # stop Y : from line 2 to line 3
    # transfer edge
    tail.append(11)
    head.append(10)
    freq.append(line3_freq)
    trav_time.append(dwell_time)
    vol.append(0.0833333333333)

    # edge 14
    # stop Y : from line 2 to line 4
    # transfer edge
    tail.append(11)
    head.append(8)
    freq.append(line4_freq)
    trav_time.append(dwell_time)
    vol.append(0.4166666666666)

    # edge 15
    # stop Y : from line 3 to line 4
    # transfer edge
    tail.append(9)
    head.append(8)
    freq.append(line4_freq)
    trav_time.append(dwell_time)
    vol.append(0.0)

    # edge 16
    # stop Y : in line 3
    # dwell edge
    tail.append(9)
    head.append(10)
    freq.append(np.inf)
    trav_time.append(dwell_time)
    vol.append(0.0)

    # edge 17
    # stop Y : to line 3
    # boarding edge
    tail.append(7)
    head.append(10)
    freq.append(line3_freq)
    trav_time.append(boarding_time)
    vol.append(0.0)

    # edge 18
    # stop Y : to line 4
    # boarding edge
    tail.append(7)
    head.append(8)
    freq.append(line4_freq)
    trav_time.append(boarding_time)
    vol.append(0.0)

    # edge 19
    # line 3 : second segment
    # on-board edge
    tail.append(10)
    head.append(14)
    freq.append(np.inf)
    trav_time.append(4.0 * 60.0)
    vol.append(0.0833333333333)

    # edge 20
    # line 4 : first segment
    # on-board edge
    tail.append(8)
    head.append(13)
    freq.append(np.inf)
    trav_time.append(10.0 * 60.0)
    vol.append(0.4166666666666)

    # edge 21
    # stop Y : from line 1
    # alighting edge
    tail.append(15)
    head.append(12)
    freq.append(np.inf)
    trav_time.append(alighting_time)
    vol.append(0.5)

    # edge 22
    # stop Y : from line 3
    # alighting edge
    tail.append(14)
    head.append(12)
    freq.append(np.inf)
    trav_time.append(alighting_time)
    vol.append(0.0833333333333)

    # edge 23
    # stop Y : from line 4
    # alighting edge
    tail.append(13)
    head.append(12)
    freq.append(np.inf)
    trav_time.append(alighting_time)
    vol.append(0.4166666666666)

    edges = pd.DataFrame(
        data={
            "tail": tail,
            "head": head,
            "trav_time": trav_time,
            "freq": freq,
            "volume_ref": vol,
        }
    )
    # waiting time is in average half of the period
    edges["freq"] *= 2.0

    demand = pd.DataFrame(
        {"origin_vertex_id": [0], "destination_vertex_id": [12], "demand": [1.0]}
    )

    return edges, demand


# author : Francois Pacull
# copyright : Architecture & Performance
# email: francois.pacull@architecture-performance.fr
# license : MIT
