"""Tests of the A* algorithm: AStar class and edsger.astar module."""

from concurrent.futures import ThreadPoolExecutor
import warnings

import numpy as np
import pandas as pd
import pytest

from edsger.astar import Workspace, compute_astar, compute_astar_coordinates
from edsger.commons import DTYPE_INF_PY
from edsger.path import AStar, Dijkstra

# finite heuristic of the vertices that cannot reach the target: any finite
# value is a lower bound of an infinite distance
LARGE = 1.0e15


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


def exact_heuristic(edges, target, **kwargs):
    """Path lengths from each vertex to the target, as a NumPy array."""
    h = Dijkstra(edges, orientation="in", **kwargs).run(target)
    h[np.isinf(h)] = LARGE
    return h


def check_path(edges, path, source, target, length, exact=True):
    """The path goes from source to target along edges, and its weight sum is
    the path length."""
    assert path[0] == source
    assert path[-1] == target
    weights = edges.groupby(["tail", "head"])["weight"].min()
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        total += weights.loc[(int(u), int(v))]
    if exact:
        assert total == length
    else:
        assert total == pytest.approx(length, rel=1e-12)


# --------------------------------------------------------- comparison with Dijkstra


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("permute", [False, True])
def test_zero_heuristic_same_as_dijkstra(seed, permute):
    edges = random_edges(seed=seed, scale=2 if permute else 1)
    astar = AStar(edges, permute=permute)
    sp = Dijkstra(edges, permute=permute)
    for source, target in random_pairs(astar.get_vertices(), 40, seed):
        expected = sp.run(int(source))[target]
        # same path, summed in the same order: exactly the same length
        assert astar.run(int(source), int(target)) == expected


@pytest.mark.parametrize("seed", range(5))
def test_exact_heuristic(seed):
    edges = random_edges(seed=seed)
    astar = AStar(edges)
    sp = Dijkstra(edges)
    for source, target in random_pairs(astar.get_vertices(), 20, seed):
        h = exact_heuristic(edges, int(target))
        expected = sp.run(int(source))[target]
        result = astar.run(int(source), int(target), heuristic=h)
        # the heuristic is exact up to the rounding errors, which can change
        # the order of near-ties
        if np.isinf(expected):
            assert np.isinf(result)
        else:
            assert result == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("seed", range(5))
def test_inconsistent_admissible_heuristic(seed):
    """A random fraction of the exact distances is admissible but not
    consistent: the vertices are scanned again, the result is still exact."""
    edges = random_edges(seed=seed)
    astar = AStar(edges)
    sp = Dijkstra(edges)
    rng = np.random.default_rng(seed)
    for source, target in random_pairs(astar.get_vertices(), 20, seed):
        h = exact_heuristic(edges, int(target))
        h = np.where(h < LARGE, 0.9 * rng.random(len(h)) * h, LARGE)
        expected = sp.run(int(source))[target]
        assert astar.run(int(source), int(target), heuristic=h) == expected


def test_not_admissible_heuristic():
    """An overestimating heuristic gives the length of a path, not
    necessarily the shortest."""
    edges = random_edges(seed=3)
    astar = AStar(edges)
    sp = Dijkstra(edges)
    rng = np.random.default_rng(0)
    for source, target in random_pairs(astar.get_vertices(), 20, 4):
        h = 100.0 * rng.random(astar.n_vertices)
        h[target] = 0.0
        length = astar.run(int(source), int(target), heuristic=h, path_tracking=True)
        expected = sp.run(int(source))[target]
        if np.isinf(expected):
            assert np.isinf(length)
            continue
        assert length >= expected
        check_path(edges, astar.get_path(), source, target, length)


@pytest.mark.parametrize("seed", range(3))
def test_integer_weights_with_zeros(seed):
    edges = random_edges(seed=seed, integer=True)
    astar = AStar(edges)
    sp = Dijkstra(edges)
    for source, target in random_pairs(astar.get_vertices(), 40, seed):
        h = exact_heuristic(edges, int(target))
        expected = sp.run(int(source))[target]
        assert astar.run(int(source), int(target)) == expected
        assert astar.run(int(source), int(target), heuristic=h) == expected


def test_dense_graph_heap_growth():
    """Many more heap entries than the initial capacity of the heap."""
    edges = random_edges(seed=7, n=150, m=20000)
    astar = AStar(edges)
    sp = Dijkstra(edges)
    for source, target in random_pairs(astar.get_vertices(), 10, 7):
        assert astar.run(int(source), int(target)) == sp.run(int(source))[target]


# --------------------------------------------------------- special cases


def test_unreachable_target():
    edges = pd.DataFrame({"tail": [0, 1, 3], "head": [1, 2, 2], "weight": [1.0] * 3})
    astar = AStar(edges)
    assert astar.run(0, 3) == np.inf
    assert astar.run(0, 3, return_inf=False) == DTYPE_INF_PY
    assert astar.run(0, 3, path_tracking=True) == np.inf
    assert astar.get_path() is None
    # the workspace is clean after an unreachable target
    assert astar.run(0, 2) == 2.0


def test_source_equals_target():
    edges = random_edges(seed=1)
    astar = AStar(edges)
    v = int(edges["tail"].iloc[0])
    assert astar.run(v, v, path_tracking=True) == 0.0
    np.testing.assert_array_equal(astar.get_path(), [v])


def test_get_path():
    edges = random_edges(seed=2)
    astar = AStar(edges)
    sp = Dijkstra(edges)
    for source, target in random_pairs(astar.get_vertices(), 30, 2):
        h = exact_heuristic(edges, int(target))
        length = astar.run(int(source), int(target), heuristic=h, path_tracking=True)
        path = astar.get_path()
        if np.isinf(length):
            assert path is None
            continue
        assert path.dtype == np.uint32
        check_path(edges, path, source, target, length)
        assert length == pytest.approx(sp.run(int(source))[target], rel=1e-12)


def test_get_path_without_tracking():
    edges = random_edges(seed=2)
    astar = AStar(edges)
    with pytest.warns(UserWarning):
        assert astar.get_path() is None
    astar.run(0, 1, path_tracking=True)
    astar.run(0, 1)
    with pytest.warns(UserWarning):
        assert astar.get_path() is None


def test_permute_negative_labels():
    edges = random_edges(seed=4, scale=3, offset=-200)
    astar = AStar(edges, permute=True)
    sp = Dijkstra(edges, permute=True)
    sp_in = Dijkstra(edges, orientation="in", permute=True)
    vertices = astar.get_vertices()
    assert vertices.min() < 0
    for source, target in random_pairs(vertices, 20, 4):
        expected = sp.run(int(source), output="pandas")[target]
        h = sp_in.run(int(target), output="pandas")
        h[np.isinf(h)] = LARGE
        length = astar.run(int(source), int(target), heuristic=h, path_tracking=True)
        assert length == pytest.approx(expected, rel=1e-12)
        if np.isinf(length):
            assert astar.get_path() is None
        else:
            check_path(edges, astar.get_path(), source, target, length, exact=False)
        assert astar.run(int(source), int(target)) == expected
    # a NumPy array cannot be indexed by negative labels
    with pytest.raises(ValueError, match="pandas Series"):
        astar.run(
            int(vertices[0]),
            int(vertices[1]),
            heuristic=np.zeros(int(vertices.max()) + 1),
        )


def test_permute_numpy_heuristic():
    """With permute=True and non-negative labels, the NumPy heuristic is
    indexed by vertex label."""
    edges = random_edges(seed=5, scale=5)
    astar = AStar(edges, permute=True)
    sp = Dijkstra(edges)
    for source, target in random_pairs(astar.get_vertices(), 20, 5):
        h = exact_heuristic(edges, int(target))
        assert len(h) == edges[["tail", "head"]].max().max() + 1
        expected = sp.run(int(source))[target]
        result = astar.run(int(source), int(target), heuristic=h)
        assert result == pytest.approx(expected, rel=1e-12)


def test_series_heuristic_with_extra_vertices():
    edges = random_edges(seed=6)
    astar = AStar(edges)
    target = int(edges["head"].iloc[0])
    h = exact_heuristic(edges, target)
    series = pd.Series(np.append(h, 0.0), index=np.arange(len(h) + 1))
    source = int(edges["tail"].iloc[0])
    assert astar.run(source, target, heuristic=series) == astar.run(
        source, target, heuristic=h
    )


def test_threads():
    """Concurrent runs on the same instance: one uses the workspace of the
    instance, the others temporary ones."""
    edges = random_edges(seed=8, n=2000, m=10000)
    astar = AStar(edges)
    pairs = random_pairs(astar.get_vertices(), 64, 8)
    expected = [astar.run(int(s), int(t)) for s, t in pairs]
    with ThreadPoolExecutor(max_workers=8) as executor:
        result = list(executor.map(lambda p: astar.run(int(p[0]), int(p[1])), pairs))
    assert result == expected


# --------------------------------------------------------- coordinates


def coordinate_graph(seed, metric, n=300, m=1500):
    rng = np.random.default_rng(seed)
    if metric == "euclidean":
        coords = pd.DataFrame({"x": rng.random(n) * 100, "y": rng.random(n) * 100})
    else:
        # longitude, latitude in degrees
        coords = pd.DataFrame({"x": 2.0 + rng.random(n), "y": 48.0 + rng.random(n)})
    tail = rng.integers(0, n, m)
    head = rng.integers(0, n, m)
    x = coords["x"].to_numpy()
    y = coords["y"].to_numpy()
    if metric == "euclidean":
        dist = np.hypot(x[head] - x[tail], y[head] - y[tail])
    else:
        lon, lat = np.radians(x), np.radians(y)
        a = (
            np.sin(0.5 * (lat[head] - lat[tail])) ** 2
            + np.cos(lat[head])
            * np.cos(lat[tail])
            * np.sin(0.5 * (lon[head] - lon[tail])) ** 2
        )
        dist = 2 * 6371008.8 * np.arcsin(np.sqrt(a))
    # travel times at a speed of at most 10
    weight = dist / 10.0 * (1.0 + rng.random(m))
    edges = pd.DataFrame({"tail": tail, "head": head, "weight": weight})
    return edges, coords


@pytest.mark.parametrize("metric", ["euclidean", "haversine"])
@pytest.mark.parametrize("seed", range(3))
def test_coordinates(metric, seed):
    edges, coords = coordinate_graph(seed, metric)
    astar = AStar(edges)
    factor = astar.set_coordinates(coords, metric=metric)
    # the weights are at least 0.1 times the distance, the largest factor is
    # slightly above 0.1
    assert 0.1 * (1 - 1e-6) <= factor < 0.11
    sp = Dijkstra(edges)
    for source, target in random_pairs(astar.get_vertices(), 30, seed):
        expected = sp.run(int(source))[target]
        length = astar.run(int(source), int(target), path_tracking=True)
        assert length == expected
        if not np.isinf(length):
            check_path(edges, astar.get_path(), source, target, length)
    # an explicit heuristic takes precedence over the coordinates
    assert astar.run(0, 1, heuristic=np.zeros(astar.n_vertices)) == sp.run(0)[1]


def test_coordinates_permute():
    edges, coords = coordinate_graph(0, "euclidean")
    edges[["tail", "head"]] = edges[["tail", "head"]] * 7 - 1000
    coords.index = coords.index * 7 - 1000
    astar = AStar(edges, permute=True)
    astar.set_coordinates(coords)
    sp = Dijkstra(edges, permute=True)
    for source, target in random_pairs(astar.get_vertices(), 30, 0):
        expected = sp.run(int(source), output="pandas")[target]
        assert astar.run(int(source), int(target)) == expected


def test_coordinates_given_factor():
    edges, coords = coordinate_graph(1, "euclidean")
    astar = AStar(edges)
    assert astar.set_coordinates(coords, factor=0.05) == 0.05
    sp = Dijkstra(edges)
    for source, target in random_pairs(astar.get_vertices(), 10, 1):
        assert astar.run(int(source), int(target)) == sp.run(int(source))[target]
    # a factor of 0 is the zero heuristic
    assert astar.set_coordinates(coords, factor=0) == 0.0
    assert astar.run(3, 4) == sp.run(3)[4]


def test_coordinates_zero_length_edges():
    """Edges between vertices with the same coordinates do not constrain the
    factor; a zero weight edge between distinct points makes it zero."""
    edges = pd.DataFrame({"tail": [0, 1], "head": [1, 2], "weight": [1.0, 2.0]})
    coords = pd.DataFrame({"x": [0.0, 0.0, 1.0], "y": [0.0, 0.0, 0.0]})
    astar = AStar(edges)
    assert astar.set_coordinates(coords) == pytest.approx(2.0)
    assert astar.run(0, 2) == 3.0
    edges = pd.DataFrame({"tail": [0, 1], "head": [1, 2], "weight": [0.0, 2.0]})
    coords = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 0.0, 0.0]})
    astar = AStar(edges)
    assert astar.set_coordinates(coords) == 0.0
    assert astar.run(0, 2) == 2.0


def test_coordinates_invalid():
    edges, coords = coordinate_graph(0, "euclidean")
    astar = AStar(edges)
    with pytest.raises(TypeError):
        astar.set_coordinates(coords.to_numpy())
    with pytest.raises(ValueError, match="metric"):
        astar.set_coordinates(coords, metric="manhattan")
    with pytest.raises(KeyError):
        astar.set_coordinates(coords, x="lon")
    with pytest.raises(ValueError, match="finite coordinates"):
        astar.set_coordinates(coords.iloc[1:])
    bad = coords.copy()
    bad.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite coordinates"):
        astar.set_coordinates(bad)
    with pytest.raises(ValueError, match="unique"):
        astar.set_coordinates(pd.concat([coords, coords]))
    with pytest.raises(ValueError, match="latitudes"):
        astar.set_coordinates(coords, metric="haversine")
    for factor in (-1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="factor"):
            astar.set_coordinates(coords, factor=factor)


# --------------------------------------------------------- invalid arguments


def test_invalid_vertices():
    edges = random_edges(seed=0, n=10, m=30)
    astar = AStar(edges)
    with pytest.raises(ValueError):
        astar.run(0, 10)
    with pytest.raises(ValueError):
        astar.run(-1, 0)
    with pytest.raises(ValueError):
        astar.run(10, 0)


def test_invalid_heuristic():
    edges = random_edges(seed=0, n=10, m=30)
    astar = AStar(edges)
    n = astar.n_vertices
    with pytest.raises(ValueError, match="one element per integer"):
        astar.run(0, 1, heuristic=np.zeros(n - 1))
    with pytest.raises(ValueError, match="one element per integer"):
        astar.run(0, 1, heuristic=np.zeros((n, 1)))
    with pytest.raises(TypeError):
        astar.run(0, 1, heuristic=np.array(["a"] * n))
    for value, match in ((np.nan, "NaN"), (-1.0, "non-negative"), (np.inf, "finite")):
        h = np.zeros(n)
        h[3] = value
        with pytest.raises(ValueError, match=match):
            astar.run(0, 1, heuristic=h)
    with pytest.raises(ValueError, match="for every vertex"):
        astar.run(0, 1, heuristic=pd.Series(np.zeros(n - 1)))
    with pytest.raises(ValueError, match="unique"):
        astar.run(0, 1, heuristic=pd.Series(np.zeros(n), index=[0] * n))
    # integer arrays are accepted
    assert astar.run(0, 1, heuristic=np.zeros(n, dtype=int)) == astar.run(0, 1)


def test_negative_weights():
    edges = pd.DataFrame({"tail": [0, 1], "head": [1, 2], "weight": [1.0, -1.0]})
    with pytest.raises(ValueError, match="nonnegative"):
        AStar(edges)
    edges.loc[1, "weight"] = np.nan
    with pytest.raises(ValueError, match="nonnegative"):
        AStar(edges)


def test_check_edges():
    edges = pd.DataFrame({"tail": [0, 1], "head": [1, 2], "w": [1.0, 1.0]})
    with pytest.raises(KeyError):
        AStar(edges, check_edges=True)
    assert AStar(edges, weight="w", check_edges=True).run(0, 2) == 2.0


# --------------------------------------------------------- Cython functions


def csr_graph():
    """0 -> 1 -> 2, 0 -> 2"""
    indptr = np.array([0, 2, 3, 3], dtype=np.uint32)
    indices = np.array([1, 2, 2], dtype=np.uint32)
    data = np.array([1.0, 3.0, 1.0])
    return indptr, indices, data


def test_cython_functions():
    indptr, indices, data = csr_graph()
    assert compute_astar(indptr, indices, data, 0, 2, 3) == 2.0
    h = np.array([2.0, 1.0, 0.0])
    h.setflags(write=False)
    workspace = Workspace(3)
    assert compute_astar(indptr, indices, data, 0, 2, 3, h, True, workspace) == 2.0
    np.testing.assert_array_equal(workspace.get_predecessor()[[0, 1, 2]], [0, 0, 1])
    assert compute_astar(indptr, indices, data, 2, 0, 3) == DTYPE_INF_PY
    x = np.array([0.0, 1.0, 2.0])
    y = np.zeros(3)
    length = compute_astar_coordinates(
        indptr, indices, data, 0, 2, 3, x, y, "euclidean", 1.0
    )
    assert length == 2.0


def test_cython_invalid_arguments():
    indptr, indices, data = csr_graph()
    for source, target in ((3, 0), (0, 3), (-1, 0), (0, -1)):
        with pytest.raises(IndexError):
            compute_astar(indptr, indices, data, source, target, 3)
    with pytest.raises(ValueError):
        compute_astar(indptr, indices, data, 0, 2, 4)
    with pytest.raises(ValueError):
        compute_astar(indptr, indices, data[:2], 0, 2, 3)
    with pytest.raises(ValueError):
        compute_astar(indptr, indices, data, 0, 2, 3, np.zeros(2))
    with pytest.raises(ValueError):
        compute_astar(indptr, indices, data, 0, 2, 3, workspace=Workspace(4))
    with pytest.raises(ValueError):
        compute_astar(indptr, indices, data, 0, 2, 3, heap_length=-1)
    with pytest.raises(ValueError):
        Workspace(0)
    x = np.zeros(3)
    with pytest.raises(ValueError):
        compute_astar_coordinates(
            indptr, indices, data, 0, 2, 3, x[:2], x, "euclidean", 1.0
        )
    with pytest.raises(ValueError):
        compute_astar_coordinates(indptr, indices, data, 0, 2, 3, x, x, "l1", 1.0)
    for factor in (-1.0, np.nan, np.inf):
        with pytest.raises(ValueError):
            compute_astar_coordinates(
                indptr, indices, data, 0, 2, 3, x, x, "euclidean", factor
            )


def test_cython_negative_weights():
    """A negative weight found by the search raises an error, even on a
    negative cycle, and the workspace is left clean."""
    # 0 -> 1 -> 2 -> 1 (negative cycle), target 3 not reachable
    indptr = np.array([0, 1, 2, 3, 3], dtype=np.uint32)
    indices = np.array([1, 2, 1], dtype=np.uint32)
    data = np.array([1.0, 1.0, -5.0])
    workspace = Workspace(4)
    with pytest.raises(ValueError, match="negative"):
        compute_astar(indptr, indices, data, 0, 3, 4, workspace=workspace)
    data[2] = 1.0
    assert compute_astar(indptr, indices, data, 0, 2, 4, workspace=workspace) == 2.0


def test_no_warning_on_normal_runs():
    edges = random_edges(seed=0)
    astar = AStar(edges)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        astar.run(0, 1, path_tracking=True)
        astar.get_path()
