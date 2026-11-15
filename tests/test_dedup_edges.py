"""Removal of the parallel edges with NumPy, against a pandas groupby."""

import numpy as np
import pandas as pd
import pytest

from edsger.path import Dijkstra, _dedup_sorted_edges


def groupby_reference(edges, weight):
    grouped = edges.groupby(["tail", "head"], as_index=False)
    if weight is None:
        return grouped.first()
    return grouped["weight"].min()


def random_edges(rng, n, n_vertices, dtype, offset=0):
    return pd.DataFrame(
        {
            "tail": rng.integers(0, n_vertices, n).astype(dtype) + dtype(offset),
            "head": rng.integers(0, n_vertices, n).astype(dtype) + dtype(offset),
            "weight": rng.integers(0, 5, n).astype(np.float64),  # many ties
        }
    )


@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize(
    "dtype, offset",
    [
        (np.int64, 0),
        (np.int32, 0),
        (np.uint32, 0),
        (np.uint64, 0),
        (np.int64, -50),  # negative labels
        (np.int64, 2**62),  # labels too large for a single int64 key
        (np.uint64, 2**63 + 5),  # labels beyond the int64 range
    ],
)
@pytest.mark.parametrize("n", [0, 1, 2, 500])
def test_same_as_groupby(weighted, dtype, offset, n):
    rng = np.random.default_rng(n)
    edges = random_edges(rng, n, 30, dtype, offset)
    if not weighted:
        edges = edges[["tail", "head"]]
    weight = "weight" if weighted else None
    result = _dedup_sorted_edges(edges, "tail", "head", weight)
    pd.testing.assert_frame_equal(result, groupby_reference(edges, weight))


def test_large_labels_spread():
    """Labels spread over the int64 range: lexsort fallback."""
    edges = pd.DataFrame(
        {
            "tail": np.array([2**62, -(2**62), 2**62, 0], dtype=np.int64),
            "head": np.array([-(2**62), 2**62, -(2**62), 1], dtype=np.int64),
            "weight": [3.0, 1.0, 2.0, 4.0],
        }
    )
    result = _dedup_sorted_edges(edges, "tail", "head", "weight")
    pd.testing.assert_frame_equal(result, groupby_reference(edges, "weight"))


def test_unweighted_keeps_first_edge_with_extra_column():
    edges = pd.DataFrame(
        {"tail": [1, 0, 1, 1], "head": [2, 1, 2, 2], "other": [10, 20, 30, 40]}
    )
    result = _dedup_sorted_edges(edges, "tail", "head", None)
    expected = pd.DataFrame({"tail": [0, 1], "head": [1, 2], "other": [20, 10]})
    pd.testing.assert_frame_equal(result, expected)


def test_sorted_unique_is_not_copied():
    edges = pd.DataFrame(
        {"tail": [0, 0, 1, 2], "head": [1, 3, 0, 0], "weight": [1.0, 2.0, 3.0, 4.0]}
    )
    assert _dedup_sorted_edges(edges, "tail", "head", "weight") is edges


def test_sorted_unique_index_is_reset():
    edges = pd.DataFrame(
        {"tail": [0, 1], "head": [1, 0], "weight": [1.0, 2.0]}, index=[5, 7]
    )
    result = _dedup_sorted_edges(edges, "tail", "head", "weight")
    assert isinstance(result.index, pd.RangeIndex)
    assert result.index.start == 0
    pd.testing.assert_frame_equal(result, groupby_reference(edges, "weight"))


def test_unsorted_index_is_reset():
    edges = pd.DataFrame(
        {"tail": [1, 0, 1], "head": [0, 1, 0], "weight": [2.0, 1.0, 0.5]},
        index=[9, 3, 4],
    )
    result = _dedup_sorted_edges(edges, "tail", "head", "weight")
    pd.testing.assert_frame_equal(result, groupby_reference(edges, "weight"))


@pytest.mark.parametrize("permute", [False, True])
def test_dijkstra_with_parallel_edges(permute):
    rng = np.random.default_rng(0)
    edges = random_edges(rng, 2000, 200, np.int64)
    sp = Dijkstra(edges, permute=permute)
    pd.testing.assert_frame_equal(sp._edges.reset_index(drop=True), sp._edges)
    expected = groupby_reference(edges, "weight")
    if not permute:
        pd.testing.assert_frame_equal(sp._edges, expected)
    assert sp._n_edges == len(expected)
