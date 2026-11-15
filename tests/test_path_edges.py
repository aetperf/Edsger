"""Tests of get_path_edges: the edges along a tracked path."""

import numpy as np
import pandas as pd
import pytest

from edsger import BFS, DFS, AStar, BellmanFord, BidirectionalDijkstra, Dijkstra


def random_edges(seed=0, n=300, m=1500, scale=1, offset=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "tail": rng.integers(0, n, m) * scale + offset,
            "head": rng.integers(0, n, m) * scale + offset,
            "weight": rng.random(m),
        }
    )


def check_chain(path_edges, start, end):
    """The edges follow each other, from start to end."""
    tails = path_edges["tail"].to_numpy()
    heads = path_edges["head"].to_numpy()
    assert tails[0] == start
    assert heads[-1] == end
    np.testing.assert_array_equal(tails[1:], heads[:-1])


def check_in_graph(path_edges, edges):
    """Each edge is in the input, with its smallest weight."""
    smallest = edges.groupby(["tail", "head"], as_index=False)["weight"].min()
    merged = path_edges.merge(smallest, on=["tail", "head"], suffixes=("", "_in"))
    assert len(merged) == len(path_edges)
    np.testing.assert_array_equal(merged["weight"], merged["weight_in"])


@pytest.mark.parametrize("cls", [Dijkstra, BellmanFord])
@pytest.mark.parametrize("orientation", ["out", "in"])
@pytest.mark.parametrize("permute", [False, True])
def test_vertex_graph_path_edges(cls, orientation, permute):
    edges = random_edges(scale=3 if permute else 1)
    sp = cls(edges, orientation=orientation, permute=permute)
    run_vertex = int(sp.get_vertices()[0])
    lengths = sp.run(run_vertex, path_tracking=True, output="pandas")
    reached = lengths[np.isfinite(lengths) & (lengths > 0)].index[:20]
    assert len(reached) > 0
    for other in reached:
        path_edges = sp.get_path_edges(int(other))
        assert list(path_edges.columns) == ["tail", "head", "weight"]
        start, end = (
            (run_vertex, other) if orientation == "out" else (other, run_vertex)
        )
        check_chain(path_edges, start, end)
        check_in_graph(path_edges, edges)
        assert path_edges["weight"].sum() == pytest.approx(lengths[other], rel=1e-12)
        # same vertices as get_path
        vertices = sp.get_path(int(other))
        if orientation == "out":
            vertices = vertices[::-1]
        expected = np.append(path_edges["tail"].to_numpy(), path_edges["head"].iloc[-1])
        np.testing.assert_array_equal(vertices, expected)


@pytest.mark.parametrize("cls", [BFS, DFS])
@pytest.mark.parametrize("orientation", ["out", "in"])
def test_unweighted_path_edges(cls, orientation):
    edges = random_edges(seed=1)[["tail", "head"]]
    search = cls(edges, orientation=orientation)
    search.run(0, path_tracking=True)
    vertices = search.get_path(5)
    assert len(vertices) > 1
    path_edges = search.get_path_edges(5)
    assert list(path_edges.columns) == ["tail", "head"]
    assert len(path_edges) == len(vertices) - 1
    start, end = (0, 5) if orientation == "out" else (5, 0)
    check_chain(path_edges, start, end)
    tails, heads, weights = search.get_path_edges(5, output="numpy")
    np.testing.assert_array_equal(tails, path_edges["tail"])
    np.testing.assert_array_equal(heads, path_edges["head"])
    np.testing.assert_array_equal(weights, np.ones(len(tails)))


@pytest.mark.parametrize("cls", [AStar, BidirectionalDijkstra])
@pytest.mark.parametrize("permute", [False, True])
def test_pair_path_edges(cls, permute):
    edges = random_edges(
        seed=2, scale=7 if permute else 1, offset=-20 if permute else 0
    )
    sp = cls(edges, permute=permute)
    vertices = sp.get_vertices()
    reference = Dijkstra(edges, permute=permute)
    for target in vertices[1:30]:
        source = int(vertices[0])
        length = sp.run(source, int(target), path_tracking=True)
        path_edges = sp.get_path_edges()
        if np.isinf(length):
            assert path_edges is None
            continue
        check_chain(path_edges, source, target)
        check_in_graph(path_edges, edges)
        assert path_edges["weight"].sum() == pytest.approx(length, rel=1e-12)
        # the bidirectional search adds the weights in another order
        assert length == pytest.approx(
            reference.run(source, output="pandas")[target], rel=1e-12
        )
        np.testing.assert_array_equal(
            sp.get_path(), np.append(path_edges["tail"], path_edges["head"].iloc[-1])
        )


def test_parallel_edges_smallest_weight():
    edges = pd.DataFrame(
        {"a": [0, 0, 1, 1], "b": [1, 1, 2, 2], "c": [3.0, 1.0, 2.0, 5.0]}
    )
    sp = Dijkstra(edges, tail="a", head="b", weight="c")
    sp.run(0, path_tracking=True)
    path_edges = sp.get_path_edges(2)
    pd.testing.assert_frame_equal(
        path_edges,
        pd.DataFrame({"a": [0, 1], "b": [1, 2], "c": [1.0, 2.0]}),
        check_dtype=False,
    )


def test_no_path_and_empty_path():
    edges = pd.DataFrame({"tail": [0, 1], "head": [1, 2], "weight": [1.0, 1.0]})
    sp = Dijkstra(edges)
    sp.run(1, path_tracking=True)
    # vertex 0 cannot be reached from 1, and 1 is the vertex of the run
    for vertex in (0, 1):
        path_edges = sp.get_path_edges(vertex)
        assert len(path_edges) == 0
        assert list(path_edges.columns) == ["tail", "head", "weight"]
    tails, heads, weights = sp.get_path_edges(0, output="numpy")
    assert len(tails) == len(heads) == len(weights) == 0

    bidir = BidirectionalDijkstra(edges)
    bidir.run(2, 0, path_tracking=True)
    assert bidir.get_path_edges() is None
    bidir.run(1, 1, path_tracking=True)
    assert len(bidir.get_path_edges()) == 0


def test_not_tracked_warns():
    edges = pd.DataFrame({"tail": [0, 1], "head": [1, 2], "weight": [1.0, 1.0]})
    sp = Dijkstra(edges)
    sp.run(0)
    with pytest.warns(UserWarning, match="path_tracking"):
        assert sp.get_path_edges(2) is None
    bidir = BidirectionalDijkstra(edges)
    with pytest.warns(UserWarning, match="path_tracking"):
        assert bidir.get_path_edges() is None
    bidir.run(0, 2)
    with pytest.warns(UserWarning, match="path_tracking"):
        assert bidir.get_path_edges() is None


def test_polars_output_and_invalid_output():
    pl = pytest.importorskip("polars")
    edges = random_edges(seed=3, offset=-100)
    sp = Dijkstra(edges, permute=True)
    lengths = sp.run(-100, path_tracking=True, output="pandas")
    target = int(lengths[np.isfinite(lengths) & (lengths > 0)].index[0])
    path_edges = sp.get_path_edges(target, output="polars")
    assert isinstance(path_edges, pl.DataFrame)
    assert path_edges["tail"][0] == -100
    with pytest.raises(ValueError, match="output"):
        sp.get_path_edges(target, output="list")
