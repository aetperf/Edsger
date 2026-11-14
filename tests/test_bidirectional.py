"""Tests of the bidirectional Dijkstra's algorithm: BidirectionalDijkstra class
and edsger.bidirectional_dijkstra module."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest

from edsger.bidirectional_dijkstra import Workspace, compute_bidirectional
from edsger.commons import DTYPE_INF_PY
from edsger.path import BidirectionalDijkstra, Dijkstra


def random_edges(seed=0, n=200, m=1000, scale=1, offset=0, integer=False):
    rng = np.random.default_rng(seed)
    weight = rng.integers(0, 10, m).astype(float) if integer else rng.random(m)
    return pd.DataFrame(
        {
            "tail": rng.integers(0, n, m) * scale + offset,
            "head": rng.integers(0, n, m) * scale + offset,
            "weight": weight,
        }
    )


def random_pairs(vertices, count, seed):
    rng = np.random.default_rng(seed)
    return rng.choice(vertices, (count, 2))


def check_path(edges, path, source, target, length):
    """The path goes from source to target along edges, and its weight sum is
    the path length, up to the summation order."""
    assert path[0] == source
    assert path[-1] == target
    weights = edges.groupby(["tail", "head"])["weight"].min()
    total = 0.0
    for u, v in zip(path[:-1], path[1:], strict=True):
        total += weights.loc[(int(u), int(v))]
    assert total == pytest.approx(length, rel=1e-12, abs=0.0)


def assert_same_length(result, expected):
    """The bidirectional path length is the sum of two half paths: it is not
    summed in the same order as by Dijkstra, so the last bits may differ."""
    if np.isinf(expected):
        assert np.isinf(result)
    else:
        assert result == pytest.approx(expected, rel=1e-12, abs=0.0)


# --------------------------------------------------------- comparison with Dijkstra


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("permute", [False, True])
def test_same_as_dijkstra(seed, permute):
    edges = random_edges(seed=seed, scale=2 if permute else 1)
    bd = BidirectionalDijkstra(edges, permute=permute)
    sp = Dijkstra(edges, permute=permute)
    for source, target in random_pairs(bd.get_vertices(), 50, seed):
        assert_same_length(
            bd.run(int(source), int(target)), sp.run(int(source))[target]
        )


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("n,m", [(200, 1000), (1000, 2500), (50, 2000)])
def test_integer_weights_exact(seed, n, m):
    """With integer weights, zeros included, the sums are exact: same lengths."""
    edges = random_edges(seed=seed, n=n, m=m, integer=True)
    bd = BidirectionalDijkstra(edges)
    sp = Dijkstra(edges)
    for source, target in random_pairs(bd.get_vertices(), 50, seed):
        assert bd.run(int(source), int(target)) == sp.run(int(source))[target]


def test_sparse_graph_many_unreachable():
    edges = random_edges(seed=3, n=2000, m=2200)
    bd = BidirectionalDijkstra(edges)
    sp = Dijkstra(edges)
    for source, target in random_pairs(bd.get_vertices(), 100, 3):
        assert_same_length(
            bd.run(int(source), int(target)), sp.run(int(source))[target]
        )


def test_dense_graph_heap_growth():
    edges = random_edges(seed=7, n=150, m=20000)
    bd = BidirectionalDijkstra(edges)
    sp = Dijkstra(edges)
    for source, target in random_pairs(bd.get_vertices(), 10, 7):
        assert_same_length(
            bd.run(int(source), int(target)), sp.run(int(source))[target]
        )


def test_grid():
    """Many equal-length paths: lengths and paths are consistent."""
    side = 15
    vertex = np.arange(side * side).reshape(side, side)
    tails = np.concatenate(
        [
            vertex[:, :-1].ravel(),
            vertex[:, 1:].ravel(),
            vertex[:-1].ravel(),
            vertex[1:].ravel(),
        ]
    )
    heads = np.concatenate(
        [
            vertex[:, 1:].ravel(),
            vertex[:, :-1].ravel(),
            vertex[1:].ravel(),
            vertex[:-1].ravel(),
        ]
    )
    edges = pd.DataFrame({"tail": tails, "head": heads, "weight": 1.0})
    bd = BidirectionalDijkstra(edges)
    for source, target in random_pairs(bd.get_vertices(), 30, 0):
        length = bd.run(int(source), int(target), path_tracking=True)
        (r0, c0), (r1, c1) = divmod(int(source), side), divmod(int(target), side)
        assert length == abs(r0 - r1) + abs(c0 - c1)
        check_path(edges, bd.get_path(), source, target, length)


# --------------------------------------------------------- special cases


def test_unreachable_target():
    edges = pd.DataFrame({"tail": [0, 1, 3], "head": [1, 2, 2], "weight": [1.0] * 3})
    bd = BidirectionalDijkstra(edges)
    assert bd.run(0, 3) == np.inf
    assert bd.run(2, 0) == np.inf
    assert bd.run(0, 3, return_inf=False) == DTYPE_INF_PY
    assert bd.run(0, 3, path_tracking=True) == np.inf
    assert bd.get_path() is None
    # the workspace is clean after an unreachable target
    assert bd.run(0, 2) == 2.0
    assert bd.run(3, 2) == 1.0


def test_source_equals_target():
    edges = random_edges(seed=1)
    bd = BidirectionalDijkstra(edges)
    v = int(edges["tail"].iloc[0])
    assert bd.run(v, v, path_tracking=True) == 0.0
    np.testing.assert_array_equal(bd.get_path(), [v])


def test_direct_edge():
    edges = pd.DataFrame({"tail": [0], "head": [1], "weight": [2.5]})
    bd = BidirectionalDijkstra(edges)
    assert bd.run(0, 1, path_tracking=True) == 2.5
    np.testing.assert_array_equal(bd.get_path(), [0, 1])


def test_get_path():
    edges = random_edges(seed=2)
    bd = BidirectionalDijkstra(edges)
    sp = Dijkstra(edges)
    for source, target in random_pairs(bd.get_vertices(), 50, 2):
        length = bd.run(int(source), int(target), path_tracking=True)
        path = bd.get_path()
        if np.isinf(length):
            assert path is None
            continue
        assert path.dtype == np.uint32
        check_path(edges, path, source, target, length)
        assert_same_length(length, sp.run(int(source))[target])


def test_get_path_without_tracking():
    edges = random_edges(seed=2)
    bd = BidirectionalDijkstra(edges)
    with pytest.warns(UserWarning):
        assert bd.get_path() is None
    bd.run(0, 1, path_tracking=True)
    bd.run(0, 1)
    with pytest.warns(UserWarning):
        assert bd.get_path() is None


def test_permute_negative_labels():
    edges = random_edges(seed=4, scale=3, offset=-200)
    bd = BidirectionalDijkstra(edges, permute=True)
    sp = Dijkstra(edges, permute=True)
    vertices = bd.get_vertices()
    assert vertices.min() < 0
    for source, target in random_pairs(vertices, 30, 4):
        expected = sp.run(int(source), output="pandas")[target]
        length = bd.run(int(source), int(target), path_tracking=True)
        assert_same_length(length, expected)
        path = bd.get_path()
        if np.isinf(length):
            assert path is None
        else:
            assert path.dtype == np.int64
            check_path(edges, path, source, target, length)


def test_threads():
    edges = random_edges(seed=8, n=2000, m=10000)
    bd = BidirectionalDijkstra(edges)
    pairs = random_pairs(bd.get_vertices(), 64, 8)
    expected = [bd.run(int(s), int(t)) for s, t in pairs]
    with ThreadPoolExecutor(max_workers=8) as executor:
        result = list(executor.map(lambda p: bd.run(int(p[0]), int(p[1])), pairs))
    assert result == expected


# --------------------------------------------------------- invalid arguments


def test_invalid_vertices():
    edges = random_edges(seed=0, n=10, m=30)
    bd = BidirectionalDijkstra(edges)
    with pytest.raises(ValueError):
        bd.run(0, 10)
    with pytest.raises(ValueError):
        bd.run(-1, 0)
    with pytest.raises(ValueError):
        bd.run(10, 0)


def test_negative_weights():
    edges = pd.DataFrame({"tail": [0, 1], "head": [1, 2], "weight": [1.0, -1.0]})
    with pytest.raises(ValueError, match="nonnegative"):
        BidirectionalDijkstra(edges)
    edges.loc[1, "weight"] = np.nan
    with pytest.raises(ValueError, match="nonnegative"):
        BidirectionalDijkstra(edges)


def test_check_edges():
    edges = pd.DataFrame({"tail": [0, 1], "head": [1, 2], "w": [1.0, 1.0]})
    with pytest.raises(KeyError):
        BidirectionalDijkstra(edges, check_edges=True)
    bd = BidirectionalDijkstra(edges, weight="w", check_edges=True)
    assert bd.run(0, 2) == 2.0


# --------------------------------------------------------- Cython function


def graph_arrays():
    """0 -> 1 -> 2, 0 -> 2, in CSR and CSC formats."""
    csr = (
        np.array([0, 2, 3, 3], dtype=np.uint32),
        np.array([1, 2, 2], dtype=np.uint32),
        np.array([1.0, 3.0, 1.0]),
    )
    csc = (
        np.array([0, 0, 1, 3], dtype=np.uint32),
        np.array([0, 0, 1], dtype=np.uint32),
        np.array([1.0, 3.0, 1.0]),
    )
    return csr, csc


def test_cython_function():
    csr, csc = graph_arrays()
    assert compute_bidirectional(*csr, *csc, 0, 2, 3)[0] == 2.0
    workspace = Workspace(3)
    length, meet = compute_bidirectional(*csr, *csc, 0, 2, 3, True, workspace)
    assert length == 2.0
    # follow the predecessors to the source, the successors to the target
    predecessor, successor = workspace.get_path_links()
    path = [meet]
    while path[0] != 0:
        path.insert(0, int(predecessor[path[0]]))
    while path[-1] != 2:
        path.append(int(successor[path[-1]]))
    assert path == [0, 1, 2]
    assert compute_bidirectional(*csr, *csc, 2, 0, 3) == (DTYPE_INF_PY, -1)
    assert compute_bidirectional(*csr, *csc, 1, 1, 3) == (0.0, 1)


def test_cython_invalid_arguments():
    csr, csc = graph_arrays()
    for source, target in ((3, 0), (0, 3), (-1, 0), (0, -1)):
        with pytest.raises(IndexError):
            compute_bidirectional(*csr, *csc, source, target, 3)
    with pytest.raises(ValueError):
        compute_bidirectional(*csr, *csc, 0, 2, 4)
    with pytest.raises(ValueError):
        compute_bidirectional(*csr, csc[0][:3], csc[1], csc[2], 0, 2, 3)
    with pytest.raises(ValueError):
        compute_bidirectional(*csr, csc[0], csc[1], csc[2][:2], 0, 2, 3)
    with pytest.raises(ValueError):
        compute_bidirectional(*csr, *csc, 0, 2, 3, workspace=Workspace(4))
    with pytest.raises(ValueError):
        compute_bidirectional(*csr, *csc, 0, 2, 3, heap_length=-1)
    with pytest.raises(ValueError):
        Workspace(0)


def test_cython_negative_weights():
    """A negative weight found by a search raises an error, even on a
    negative cycle, and the workspace is left clean."""
    # 0 -> 1 -> 0 (negative cycle), 1 -> 2 -> 3 -> 4
    csr = (
        np.array([0, 1, 3, 4, 5, 5], dtype=np.uint32),
        np.array([1, 0, 2, 3, 4], dtype=np.uint32),
        np.array([1.0, -5.0, 1.0, 1.0, 1.0]),
    )
    csc = (
        np.array([0, 1, 2, 3, 4, 5], dtype=np.uint32),
        np.array([1, 0, 1, 2, 3], dtype=np.uint32),
        np.array([-5.0, 1.0, 1.0, 1.0, 1.0]),
    )
    workspace = Workspace(5)
    with pytest.raises(ValueError, match="negative"):
        compute_bidirectional(*csr, *csc, 0, 4, 5, workspace=workspace)
    csr[2][1] = 1.0
    csc[2][0] = 1.0
    length, _ = compute_bidirectional(*csr, *csc, 0, 4, 5, workspace=workspace)
    assert length == 4.0
