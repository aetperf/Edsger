"""Tests for dijkstra.pyx.

py.test tests/test_dijkstra.py
"""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_array
from scipy.sparse.csgraph import dijkstra as scipy_dijkstra

from edsger.dijkstra import (
    compute_sssp_01,
    compute_sssp_02,
    compute_stsp_01,
    compute_stsp_02,
)
from edsger.path import Dijkstra


def test_compute_sssp_01():
    compute_sssp_01()


def test_compute_stsp_01():
    compute_stsp_01()


def test_compute_sssp_02():
    compute_sssp_02()


def test_compute_stsp_02():
    compute_stsp_02()


@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("heap_length_ratio", [1.0, 0.01])
def test_dense_graph_heap_growth(lazy, heap_length_ratio):
    """On a dense graph, the lazy heap holds several entries per vertex and
    exceeds the vertex count: the heap must grow instead of overflowing."""
    rng = np.random.default_rng(124)
    n, degree = 300, 150
    edges = pd.DataFrame(
        {
            "tail": np.repeat(np.arange(n), degree),
            "head": rng.integers(0, n, n * degree),
            "weight": rng.random(n * degree) * 100.0,
        }
    )
    edges = edges.groupby(["tail", "head"], as_index=False).weight.min()
    graph = csr_array(
        (edges["weight"].values, (edges["tail"].values, edges["head"].values)),
        shape=(n, n),
    )
    path_lengths_ref = scipy_dijkstra(graph, directed=True, indices=0)

    sp = Dijkstra(edges, orientation="out", lazy=lazy)
    path_lengths = sp.run(vertex_idx=0, heap_length_ratio=heap_length_ratio)

    np.testing.assert_array_equal(path_lengths, path_lengths_ref)
