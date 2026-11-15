"""Tests of pickling: a built graph saved and loaded gives the same results."""

import pickle
import threading

import numpy as np
import pandas as pd
import pytest

import edsger
from edsger import (
    BFS,
    DFS,
    AStar,
    BellmanFord,
    BidirectionalDijkstra,
    Centrality,
    ConnectedComponents,
    Dijkstra,
    HyperpathGenerating,
    MinimumSpanningTree,
)


def random_edges(seed=0, n=200, m=1000):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "tail": rng.integers(0, n, m) * 3 - 30,
            "head": rng.integers(0, n, m) * 3 - 30,
            "weight": rng.random(m),
        }
    )


def roundtrip(obj, tmp_path):
    """Save to a file and load back."""
    path = tmp_path / "graph.pickle"
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(path, "rb") as f:
        return pickle.load(f)


@pytest.mark.parametrize("cls", [Dijkstra, BellmanFord, BFS, DFS])
@pytest.mark.parametrize("orientation", ["out", "in"])
def test_vertex_graphs(cls, orientation, tmp_path):
    edges = random_edges()
    graph = cls(edges, orientation=orientation, permute=True)
    loaded = roundtrip(graph, tmp_path)
    assert repr(loaded) == repr(graph)
    expected = graph.run(-30, path_tracking=True, output="pandas")
    result = loaded.run(-30, path_tracking=True, output="pandas")
    pd.testing.assert_series_equal(result, expected)
    vertex = int(expected.index[5])
    np.testing.assert_array_equal(loaded.get_path(vertex), graph.get_path(vertex))


@pytest.mark.parametrize("lazy", [True, False])
def test_dijkstra_variant(lazy, tmp_path):
    graph = Dijkstra(random_edges(), lazy=lazy, permute=True)
    loaded = roundtrip(graph, tmp_path)
    assert loaded.lazy is lazy
    np.testing.assert_array_equal(
        loaded.run_many([-30, -27], output="pandas"),
        graph.run_many([-30, -27], output="pandas"),
    )


def test_last_run_is_kept(tmp_path):
    graph = Dijkstra(random_edges(), permute=True)
    graph.run(-30, path_tracking=True, output="pandas")
    loaded = roundtrip(graph, tmp_path)
    np.testing.assert_array_equal(loaded.path_links, graph.path_links)
    pd.testing.assert_frame_equal(loaded.get_path_edges(0), graph.get_path_edges(0))


@pytest.mark.parametrize("cls", [AStar, BidirectionalDijkstra])
def test_pair_graphs(cls, tmp_path):
    edges = random_edges(seed=1)
    graph = cls(edges, permute=True)
    # pickled before and after a run, which creates the workspace
    for _ in range(2):
        loaded = roundtrip(graph, tmp_path)
        assert loaded._workspace is None
        assert isinstance(loaded._workspace_lock, type(threading.Lock()))
        for target in (0, 30, 90):
            assert loaded.run(-30, target, path_tracking=True) == graph.run(
                -30, target, path_tracking=True
            )
            np.testing.assert_array_equal(loaded.get_path(), graph.get_path())


def test_astar_coordinates(tmp_path):
    edges = pd.DataFrame(
        {"tail": [0, 1, 0], "head": [1, 2, 2], "weight": [1.0, 1.0, 3.0]}
    )
    graph = AStar(edges)
    graph.set_coordinates(pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 0.0, 0.0]}))
    loaded = roundtrip(graph, tmp_path)
    assert loaded.run(0, 2, path_tracking=True) == 2.0
    np.testing.assert_array_equal(loaded.get_path(), [0, 1, 2])


def test_other_classes(tmp_path):
    edges = random_edges(seed=2)
    components = roundtrip(ConnectedComponents(edges, permute=True), tmp_path)
    pd.testing.assert_series_equal(
        components.run("strong", output="pandas"),
        ConnectedComponents(edges, permute=True).run("strong", output="pandas"),
    )
    tree = roundtrip(MinimumSpanningTree(edges, permute=True), tmp_path)
    pd.testing.assert_frame_equal(
        tree.run(), MinimumSpanningTree(edges, permute=True).run()
    )
    centrality = roundtrip(Centrality(edges, permute=True), tmp_path)
    np.testing.assert_array_equal(
        centrality.pagerank(output="pandas"),
        Centrality(edges, permute=True).pagerank(output="pandas"),
    )


def test_hyperpath(tmp_path):
    edges = pd.DataFrame(
        {
            "tail": [0, 0, 1],
            "head": [1, 2, 2],
            "trav_time": [1.0, 3.0, 1.0],
            "freq": [0.1, 0.2, 0.5],
        }
    )
    graph = HyperpathGenerating(edges)
    graph.run(origin=0, destination=2, volume=1.0)
    loaded = roundtrip(graph, tmp_path)
    pd.testing.assert_frame_equal(loaded._edges, graph._edges)
    loaded.run(origin=0, destination=2, volume=1.0)
    pd.testing.assert_frame_equal(loaded._edges, graph._edges)


def test_other_version_warns(monkeypatch):
    data = pickle.dumps(Dijkstra(random_edges(), permute=True))
    monkeypatch.setattr(edsger.path, "__version__", "0.0.1")
    with pytest.warns(
        UserWarning, match="pickled with edsger .* loaded with edsger 0.0.1"
    ):
        loaded = pickle.loads(data)
    assert loaded.n_vertices > 0
