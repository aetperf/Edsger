"""Tests for the depth-first search.

py.test tests/test_dfs.py -v
"""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import depth_first_order

from edsger import dfs as dfs_module
from edsger.dfs import dfs_csc, dfs_csr
from edsger.path import DFS

SENTINEL = -9999


def random_edges(n_vertices, n_edges, seed, offset=0):
    """Random directed edges, with parallel edges and loops."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "tail": rng.integers(0, n_vertices, n_edges) + offset,
            "head": rng.integers(0, n_vertices, n_edges) + offset,
        }
    )


def scipy_dfs(tails, heads, n_vertices, start, orientation="out"):
    """Discovery order and predecessors given by SciPy, on internal indices."""
    if orientation == "in":
        tails, heads = heads, tails
    matrix = coo_matrix(
        (np.ones(len(tails)), (tails, heads)), shape=(n_vertices, n_vertices)
    ).tocsr()
    matrix.sum_duplicates()
    matrix.sort_indices()
    order, predecessors = depth_first_order(matrix, start, directed=True)
    predecessors = np.where(predecessors < 0, SENTINEL, predecessors)
    return order, predecessors.astype(np.int32)


# ---------------------------------------------------------------- Cython module


def test_cython_tests():
    dfs_module.test_dfs_csr_01()
    dfs_module.test_dfs_csc_01()


def test_kernel_invalid_arguments():
    indptr = np.array([0, 1, 2, 2], dtype=np.uint32)
    indices = np.array([1, 2], dtype=np.uint32)
    for kernel in (dfs_csr, dfs_csc):
        with pytest.raises(IndexError):
            kernel(indptr, indices, 3, 3)
        with pytest.raises(IndexError):
            kernel(indptr, indices, -1, 3)
        with pytest.raises(ValueError):
            kernel(indptr, indices, 0, 4)
        with pytest.raises(ValueError):
            kernel(indptr, indices[:1], 0, 3)
        with pytest.raises(ValueError):
            kernel(indptr[:0], indices, 0, 0)


# ---------------------------------------------------------------- DFS class


def test_simple_graph():
    edges = pd.DataFrame({"tail": [0, 0, 1, 2], "head": [1, 2, 3, 3]})
    dfs = DFS(edges)
    predecessors = dfs.run(0, path_tracking=True)
    np.testing.assert_array_equal(predecessors, [SENTINEL, 0, 0, 1])
    assert predecessors.dtype == np.int32
    np.testing.assert_array_equal(dfs.get_order(), [0, 1, 3, 2])
    np.testing.assert_array_equal(dfs.get_path(3), [3, 1, 0])
    np.testing.assert_array_equal(dfs.get_vertices(), [0, 1, 2, 3])
    assert dfs.UNREACHABLE == SENTINEL


def test_visiting_order_is_by_identifier():
    """The neighbours are visited in increasing order, whatever the edge order."""
    edges = pd.DataFrame({"tail": [0, 0, 0, 3], "head": [3, 2, 1, 1]})
    dfs = DFS(edges)
    np.testing.assert_array_equal(dfs.run(0), [SENTINEL, 0, 0, 0])
    np.testing.assert_array_equal(dfs.get_order(), [0, 1, 2, 3])


def test_orientation_in():
    edges = pd.DataFrame({"tail": [0, 0, 1, 2], "head": [1, 2, 3, 3]})
    dfs = DFS(edges, orientation="in")
    successors = dfs.run(3, path_tracking=True)
    np.testing.assert_array_equal(successors, [1, 3, 3, SENTINEL])
    np.testing.assert_array_equal(dfs.get_order(), [3, 1, 0, 2])
    np.testing.assert_array_equal(dfs.get_path(0), [0, 1, 3])
    assert dfs.orientation == "in"


@pytest.mark.parametrize("orientation", ["out", "in"])
@pytest.mark.parametrize("seed", range(10))
def test_against_scipy(orientation, seed):
    n_vertices = 200
    edges = random_edges(n_vertices, 400, seed)
    # all the vertices are in the graph
    edges = pd.concat(
        [edges, pd.DataFrame({"tail": [n_vertices - 1], "head": [0]})],
        ignore_index=True,
    )
    dfs = DFS(edges, orientation=orientation)
    assert dfs.n_vertices == n_vertices
    rng = np.random.default_rng(seed)
    for start in rng.integers(0, n_vertices, 5):
        order, predecessors = scipy_dfs(
            edges["tail"].values,
            edges["head"].values,
            n_vertices,
            start,
            orientation,
        )
        np.testing.assert_array_equal(dfs.run(start), predecessors)
        np.testing.assert_array_equal(dfs.get_order(), order)


@pytest.mark.parametrize("orientation", ["out", "in"])
def test_permute_negative_labels(orientation):
    """With permute=True, the labels can be negative and sparse: the result is
    the one on the renumbered graph, with the original labels."""
    edges = random_edges(100, 300, 3)
    labels = np.sort(np.random.default_rng(0).choice(10**6, 100, replace=False))
    labels -= 500_000
    edges_labelled = pd.DataFrame(
        {"tail": labels[edges["tail"]], "head": labels[edges["head"]]}
    )
    vertices = np.unique(np.concatenate([edges["tail"], edges["head"]]))
    positions = np.searchsorted(vertices, edges[["tail", "head"]].values)
    start = int(positions[0, 0])
    order, predecessors = scipy_dfs(
        positions[:, 0], positions[:, 1], len(vertices), start, orientation
    )

    dfs = DFS(edges_labelled, orientation=orientation, permute=True)
    vertex_labels = labels[vertices]
    for output in ["pandas", "polars"]:
        if output == "polars":
            pytest.importorskip("polars")
        result = dfs.run(int(vertex_labels[start]), output=output)
        if output == "polars":
            result = result.to_pandas().set_index("vertex_idx")["predecessor"]
        assert result.name == "predecessor"
        assert result.index.name == "vertex_idx"
        np.testing.assert_array_equal(result.index, vertex_labels)
        expected = np.where(
            predecessors == SENTINEL,
            SENTINEL,
            vertex_labels[np.maximum(predecessors, 0)],
        )
        np.testing.assert_array_equal(result.values, expected)
        np.testing.assert_array_equal(dfs.get_order(), vertex_labels[order])

    with pytest.raises(ValueError, match="negative vertex labels"):
        dfs.run(int(vertex_labels[start]))


def test_permute_numpy_output():
    edges = pd.DataFrame({"tail": [10, 10, 30], "head": [30, 20, 40]})
    dfs = DFS(edges, permute=True)
    result = dfs.run(10, path_tracking=True)
    assert len(result) == 41
    assert result[20] == 10 and result[30] == 10 and result[40] == 30
    assert (np.delete(result, [20, 30, 40]) == SENTINEL).all()
    np.testing.assert_array_equal(dfs.get_order(), [10, 20, 30, 40])
    assert dfs.get_order().dtype == np.uint32
    np.testing.assert_array_equal(dfs.get_path(40), [40, 30, 10])
    # same results without permute
    result_no_permute = DFS(edges).run(10)
    np.testing.assert_array_equal(result, result_no_permute)


@pytest.mark.parametrize("output", ["numpy", "pandas", "polars"])
def test_outputs(output):
    if output == "polars":
        pl = pytest.importorskip("polars")
    edges = pd.DataFrame({"tail": [0, 1, 3], "head": [1, 2, 4]})
    result = DFS(edges).run(0, output=output)
    expected = [SENTINEL, 0, 1, SENTINEL, SENTINEL]
    if output == "numpy":
        np.testing.assert_array_equal(result, expected)
    elif output == "pandas":
        assert isinstance(result, pd.Series)
        np.testing.assert_array_equal(result.values, expected)
        np.testing.assert_array_equal(result.index, np.arange(5))
    else:
        assert isinstance(result, pl.DataFrame)
        assert result.columns == ["vertex_idx", "predecessor"]
        np.testing.assert_array_equal(result["predecessor"].to_numpy(), expected)
    series = DFS(edges).run(0, return_series=True)
    assert isinstance(series, pd.Series)


def test_polars_input():
    pl = pytest.importorskip("polars")
    edges = pl.DataFrame({"tail": [0, 0, 1], "head": [1, 2, 3]})
    dfs = DFS(edges, check_edges=True)
    np.testing.assert_array_equal(dfs.run(0), [SENTINEL, 0, 0, 1])


def test_single_edge_and_loop():
    dfs = DFS(pd.DataFrame({"tail": [0], "head": [0]}))
    np.testing.assert_array_equal(dfs.run(0), [SENTINEL])
    np.testing.assert_array_equal(dfs.get_order(), [0])
    dfs = DFS(pd.DataFrame({"tail": [1], "head": [0]}))
    np.testing.assert_array_equal(dfs.run(0), [SENTINEL, SENTINEL])
    np.testing.assert_array_equal(dfs.get_order(), [0])


def test_long_path():
    """A path deeper than any call stack."""
    n_vertices = 300_000
    tails = np.arange(n_vertices - 1)
    edges = pd.DataFrame({"tail": tails, "head": tails + 1})
    dfs = DFS(edges)
    predecessors = dfs.run(0)
    np.testing.assert_array_equal(predecessors[1:], tails)
    np.testing.assert_array_equal(dfs.get_order(), np.arange(n_vertices))


def test_custom_columns_and_sentinel():
    edges = pd.DataFrame({"u": [0, 1], "v": [1, 2], "w": [1.0, 2.0]})
    dfs = DFS(edges, tail="u", head="v", check_edges=True, sentinel=-1)
    np.testing.assert_array_equal(dfs.run(0), [-1, 0, 1])
    assert dfs.UNREACHABLE == -1


def test_invalid_arguments():
    edges = pd.DataFrame({"tail": [0, 1], "head": [1, 2]})
    dfs = DFS(edges)
    with pytest.raises(ValueError, match="not found"):
        dfs.run(3)
    with pytest.raises(ValueError, match="non-negative"):
        dfs.run(-1)
    with pytest.raises(ValueError, match="output"):
        dfs.run(0, output="arrow")
    with pytest.raises(ValueError):
        DFS(edges, orientation="both")
    with pytest.raises(ValueError, match="sentinel must be negative"):
        DFS(edges, sentinel=0)
    with pytest.raises(TypeError, match="sentinel must be an integer"):
        DFS(edges, sentinel=-1.5)
    with pytest.raises(ValueError, match="int32"):
        DFS(edges, sentinel=-(2**40))
    with pytest.raises(ValueError, match="also a vertex label"):
        DFS(pd.DataFrame({"tail": [-1], "head": [0]}), permute=True, sentinel=-1)
    with pytest.raises(KeyError):
        DFS(edges, tail="from", check_edges=True)
    with pytest.raises(ValueError, match="not found"):
        DFS(edges, permute=True).run(7)


def test_get_order_and_path_before_run():
    dfs = DFS(pd.DataFrame({"tail": [0], "head": [1]}))
    with pytest.warns(UserWarning):
        assert dfs.get_order() is None
    dfs.run(0)
    with pytest.warns(UserWarning):
        assert dfs.get_path(1) is None
