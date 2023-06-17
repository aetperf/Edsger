"""This module makes it easy to execute common tasks in Python scripts such as generate random 
graphs.
"""
import numpy as np
import pandas as pd


def generate_random_network(n_edges=100, n_verts=20, seed=124, sort=True):
    rng = np.random.default_rng(seed=seed)
    tail = rng.integers(low=0, high=n_verts, size=n_edges)
    head = rng.integers(low=0, high=n_verts, size=n_edges)
    weight = rng.random(size=n_edges)
    edges = pd.DataFrame(data={"tail": tail, "head": head, "weight": weight})
    if sort:
        edges.sort_values(by=["tail", "head"], inplace=True)
        edges.reset_index(drop=True, inplace=True)
    return edges
