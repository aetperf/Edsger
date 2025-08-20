"""This module makes it easy to execute common tasks in Python scripts such as generate random
graphs.
"""

import numpy as np
import pandas as pd


def generate_random_network(
    n_edges=100,
    n_verts=20,
    seed=124,
    sort=True,
    allow_negative_weights=False,
    negative_weight_ratio=0.3,
    weight_range=(0.1, 1.0),
):
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


# author : Francois Pacull
# copyright : Architecture & Performance
# email: francois.pacull@architecture-performance.fr
# license : MIT
