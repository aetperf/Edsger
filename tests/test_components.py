"""Tests for the connected components.

py.test tests/test_components.py -v
"""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from edsger import components_core
from edsger.components import ConnectedComponents
from edsger.components_core import strong_components_csr, weak_components_csr


def random_edges(n_vertices, n_edges, seed):
    """Random directed edges, with parallel edges and loops."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "tail": rng.integers(0, n_vertices, n_edges),
            "head": rng.integers(0, n_vertices, n_edges),
        }
    )


def canonical(labels):
    """Renumber the labels in order of the smallest vertex of each component."""
    _, first = np.unique(labels, return_index=True)
    order = np.argsort(first)
    mapping = np.empty(len(first), dtype=np.int64)
    mapping[np.unique(labels)[order]] = np.arange(len(first))
    return mapping[np.searchsorted(np.unique(labels), labels)]


def scipy_components(tails, heads, n_vertices, connection):
    matrix = coo_matrix(
        (np.ones(len(tails)), (tails, heads)), shape=(n_vertices, n_vertices)
    ).tocsr()
    n_components, labels = connected_components(
        matrix, directed=True, connection=connection
    )
    return n_components, canonical(labels)


# ---------------------------------------------------------------- Cython module


def test_cython_tests():
    components_core.test_components_01()


@pytest.mark.parametrize("kernel", [weak_components_csr, strong_components_csr])
def test_kernel_invalid_arguments(kernel):
    indptr = np.array([0, 1, 2, 2], dtype=np.uint32)
    indices = np.array([1, 2], dtype=np.uint32)
    with pytest.raises(ValueError):
        kernel(indptr, indices, 4)
    with pytest.raises(ValueError):
        kernel(indptr, indices[:1], 3)
    with pytest.raises(ValueError):
        kernel(indptr[:0], indices, 0)


# ---------------------------------------------------------------- class


def test_simple_graph():
    # 0 <-> 1 <- 2, 3 -> 4
    edges = pd.DataFrame({"tail": [0, 1, 2, 3], "head": [1, 0, 1, 4]})
    cc = ConnectedComponents(edges)
    assert cc.n_components is None
    weak = cc.run()
    assert weak.dtype == np.int32
    np.testing.assert_array_equal(weak, [0, 0, 0, 1, 1])
    assert cc.n_components == 2
    np.testing.assert_array_equal(cc.run("strong"), [0, 0, 1, 2, 3])
    assert cc.n_components == 4


@pytest.mark.parametrize("connection", ["weak", "strong"])
@pytest.mark.parametrize("seed", range(20))
def test_against_scipy(connection, seed):
    rng = np.random.default_rng(seed)
    n_vertices = int(rng.integers(1, 300))
    n_edges = int(rng.integers(1, 3 * n_vertices + 2))
    edges = random_edges(n_vertices, n_edges, seed)
    cc = ConnectedComponents(edges, permute=True)
    vertices = cc.get_vertices()
    positions = np.searchsorted(vertices, edges[["tail", "head"]].values)
    n_components, labels = scipy_components(
        positions[:, 0], positions[:, 1], len(vertices), connection
    )
    result = cc.run(connection, output="pandas")
    np.testing.assert_array_equal(result.index, vertices)
    np.testing.assert_array_equal(result.values, labels)
    assert cc.n_components == n_components
    # same labels without permute, the integers that are not vertices get -1
    cc = ConnectedComponents(edges)
    dense = cc.run(connection)
    assert cc.n_components == n_components
    np.testing.assert_array_equal(dense[vertices], labels)
    assert (np.delete(dense, vertices) == -1).all()


@pytest.mark.parametrize("connection", ["weak", "strong"])
def test_larger_graph_against_scipy(connection):
    """Larger and sparser: many components of various sizes."""
    n_vertices = 20_000
    edges = random_edges(n_vertices, 25_000, 42)
    cc = ConnectedComponents(edges, permute=True)
    vertices = cc.get_vertices()
    positions = np.searchsorted(vertices, edges[["tail", "head"]].values)
    n_components, labels = scipy_components(
        positions[:, 0], positions[:, 1], len(vertices), connection
    )
    result = cc.run(connection, output="pandas")
    assert cc.n_components == n_components
    np.testing.assert_array_equal(result.values, labels)


def test_against_networkx():
    nx = pytest.importorskip("networkx")
    edges = random_edges(500, 700, 7)
    graph = nx.DiGraph()
    graph.add_edges_from(edges.itertuples(index=False))
    cc = ConnectedComponents(edges)
    for connection, function in [
        ("weak", nx.weakly_connected_components),
        ("strong", nx.strongly_connected_components),
    ]:
        labels = cc.run(connection, output="pandas")
        expected = sorted(sorted(c) for c in function(graph))
        groups = labels[labels >= 0].groupby(labels[labels >= 0]).groups
        found = sorted(sorted(g) for g in groups.values())
        assert found == expected


def test_labels_do_not_depend_on_edge_order():
    edges = random_edges(300, 400, 11)
    shuffled = edges.sample(frac=1.0, random_state=0).reset_index(drop=True)
    for connection in ["weak", "strong"]:
        np.testing.assert_array_equal(
            ConnectedComponents(edges).run(connection),
            ConnectedComponents(shuffled).run(connection),
        )


def test_permute_negative_labels():
    pl = pytest.importorskip("polars")
    # -5 <-> 3 -> 100, 7 -> 7
    edges = pd.DataFrame({"tail": [-5, 3, 3, 7], "head": [3, -5, 100, 7]})
    cc = ConnectedComponents(edges, permute=True)
    result = cc.run("strong", output="polars")
    assert isinstance(result, pl.DataFrame)
    assert result.columns == ["vertex_idx", "component"]
    np.testing.assert_array_equal(result["vertex_idx"].to_numpy(), [-5, 3, 7, 100])
    np.testing.assert_array_equal(result["component"].to_numpy(), [0, 0, 1, 2])
    assert cc.n_components == 3
    series = cc.run("weak", return_series=True)
    assert series.name == "component"
    assert series.index.name == "vertex_idx"
    np.testing.assert_array_equal(series.values, [0, 0, 1, 0])
    assert cc.n_components == 2
    with pytest.raises(ValueError, match="negative vertex labels"):
        cc.run()


def test_permute_numpy_output():
    edges = pd.DataFrame({"tail": [10, 30], "head": [20, 40]})
    cc = ConnectedComponents(edges, permute=True)
    result = cc.run()
    assert len(result) == 41
    np.testing.assert_array_equal(result[[10, 20, 30, 40]], [0, 0, 1, 1])
    assert (np.delete(result, [10, 20, 30, 40]) == -1).all()
    np.testing.assert_array_equal(result, ConnectedComponents(edges).run())


@pytest.mark.parametrize("output", ["numpy", "pandas", "polars"])
def test_outputs(output):
    if output == "polars":
        pl = pytest.importorskip("polars")
    edges = pd.DataFrame({"tail": [0, 2], "head": [1, 3]})
    result = ConnectedComponents(edges).run(output=output)
    if output == "numpy":
        np.testing.assert_array_equal(result, [0, 0, 1, 1])
    elif output == "pandas":
        assert isinstance(result, pd.Series)
        np.testing.assert_array_equal(result.values, [0, 0, 1, 1])
        np.testing.assert_array_equal(result.index, np.arange(4))
    else:
        assert isinstance(result, pl.DataFrame)
        np.testing.assert_array_equal(result["component"].to_numpy(), [0, 0, 1, 1])


def test_polars_input():
    pl = pytest.importorskip("polars")
    edges = pl.DataFrame({"tail": [0, 1, 2], "head": [1, 0, 2]})
    cc = ConnectedComponents(edges, check_edges=True)
    np.testing.assert_array_equal(cc.run("strong"), [0, 0, 1])


def test_small_graphs():
    cc = ConnectedComponents(pd.DataFrame({"tail": [0], "head": [0]}))
    np.testing.assert_array_equal(cc.run("strong"), [0])
    assert cc.n_components == 1
    cc = ConnectedComponents(pd.DataFrame({"tail": [1], "head": [0]}))
    np.testing.assert_array_equal(cc.run("strong"), [0, 1])
    np.testing.assert_array_equal(cc.run("weak"), [0, 0])
    # no edge: no vertex
    empty = pd.DataFrame({"tail": [], "head": []}, dtype=np.int64)
    cc = ConnectedComponents(empty)
    assert len(cc.run()) == 0
    assert cc.n_components == 0
    assert len(cc.run("strong", output="pandas")) == 0


def test_long_cycle():
    """A cycle deeper than any call stack: one strong component."""
    n_vertices = 300_000
    tails = np.arange(n_vertices)
    edges = pd.DataFrame({"tail": tails, "head": (tails + 1) % n_vertices})
    cc = ConnectedComponents(edges)
    assert (cc.run("strong") == 0).all()
    assert cc.n_components == 1
    # a path: one strong component per vertex
    cc = ConnectedComponents(edges.iloc[:-1])
    np.testing.assert_array_equal(cc.run("strong"), tails)
    assert cc.n_components == n_vertices


def test_invalid_arguments():
    edges = pd.DataFrame({"tail": [0], "head": [1]})
    cc = ConnectedComponents(edges)
    with pytest.raises(ValueError, match="connection"):
        cc.run("both")
    with pytest.raises(ValueError, match="output"):
        cc.run(output="arrow")
    with pytest.raises(ValueError):
        cc.run(output="numpy", return_series=True)
    with pytest.raises(KeyError):
        ConnectedComponents(edges, head="to", check_edges=True)
    with pytest.raises(TypeError):
        ConnectedComponents(
            pd.DataFrame({"tail": [0.5], "head": [1.0]}), check_edges=True
        )
    with pytest.raises(ValueError):
        ConnectedComponents(pd.DataFrame({"tail": [-1], "head": [1]}))
