"""This module makes it easy to execute common tasks in Python scripts such as generate random
graphs.
"""

import numpy as np
import pandas as pd


def generate_random_network(n_edges=100, n_verts=20, seed=124, sort=True):
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

    Returns
    -------
    edges : pandas.DataFrame
        A DataFrame containing the edges of the network with columns 'tail', 'head', and 'weight'.

    Examples
    --------
    >>> generate_random_network(n_edges=5, n_verts=3, seed=42)
       tail  head    weight
    0     0     2  0.975622
    1     1     0  0.128114
    2     1     0  0.450386
    3     1     2  0.786064
    4     2     0  0.761140


    Notes
    -----
    The 'tail' and 'head' columns represent the source and destination vertices of each edge,
    respectively. The 'weight' column represents the weight of each edge, which is a random
    float between 0 and 1.

    If `sort` is True, the DataFrame is sorted by the 'tail' and 'head' columns and the index
    is reset.
    """
    rng = np.random.default_rng(seed=seed)
    tail = rng.integers(low=0, high=n_verts, size=n_edges)
    head = rng.integers(low=0, high=n_verts, size=n_edges)
    weight = rng.random(size=n_edges)
    edges = pd.DataFrame(data={"tail": tail, "head": head, "weight": weight})
    if sort:
        edges.sort_values(by=["tail", "head"], inplace=True)
        edges.reset_index(drop=True, inplace=True)
    return edges


# author : Francois Pacull
# copyright : Architecture & Performance
# email: francois.pacull@architecture-performance.fr
# license : MIT
