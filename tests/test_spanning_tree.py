"""Tests for the minimum spanning tree (edsger.spanning_tree, edsger.kruskal)."""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree

from edsger.kruskal import kruskal
from edsger.spanning_tree import MinimumSpanningTree


def random_edges(n_vertices, n_edges, seed, low=1.0, high=10.0):
    """Random directed edges, with parallel and antiparallel edges and
    self-loops, and positive weights."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "tail": rng.integers(0, n_vertices, n_edges),
            "head": rng.integers(0, n_vertices, n_edges),
            "weight": rng.uniform(low, high, n_edges),
        }
    )


def scipy_mst_weight(edges, n_vertices):
    """Total weight of a minimum spanning forest with SciPy.

    SciPy reads the matrix as an undirected graph, with Kruskal's algorithm on
    all its non-zero entries, so A[i, j] and A[j, i] are both candidates for the
    pair (i, j). The duplicate (i, j) entries of a COO matrix are summed when
    converted: the parallel edges are reduced to their minimum weight first,
    as in edsger. The weights must be positive: 0 means no edge for SciPy.
    """
    reduced = edges.groupby(["tail", "head"], as_index=False)["weight"].min()
    reduced = reduced[reduced["tail"] != reduced["head"]]
    matrix = coo_matrix(
        (reduced["weight"], (reduced["tail"], reduced["head"])),
        shape=(n_vertices, n_vertices),
    ).tocsr()
    return minimum_spanning_tree(matrix).sum()


def undirected_components(edges, n_vertices):
    """Number of connected components among the vertices used by the edges."""
    matrix = coo_matrix(
        (np.ones(len(edges)), (edges["tail"], edges["head"])),
        shape=(n_vertices, n_vertices),
    ).tocsr()
    _, labels = connected_components(matrix, directed=False)
    used = np.union1d(edges["tail"], edges["head"])
    return len(np.unique(labels[used]))


def check_forest(edges, forest, weight="weight"):
    """The forest edges are input edges, and they form no cycle."""
    keys = set(zip(edges["tail"], edges["head"], edges[weight]))
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            x = parent[x]
        return x

    for t, h, w in zip(forest["tail"], forest["head"], forest[weight]):
        assert (t, h, w) in keys
        rt, rh = find(t), find(h)
        assert rt != rh, "cycle in the forest"
        parent[rt] = rh


def test_small():
    edges = pd.DataFrame(
        {
            "tail": [0, 1, 0, 2],
            "head": [1, 2, 2, 3],
            "weight": [1.0, 2.0, 4.0, 3.0],
        }
    )
    mst = MinimumSpanningTree(edges)
    assert mst.total_weight is None and mst.n_trees is None
    forest = mst.run()
    assert isinstance(forest, pd.DataFrame)
    assert list(forest.columns) == ["tail", "head", "weight"]
    assert forest["tail"].tolist() == [0, 1, 2]
    assert forest["head"].tolist() == [1, 2, 3]
    assert forest["weight"].tolist() == [1.0, 2.0, 3.0]
    assert mst.total_weight == 6.0
    assert mst.n_trees == 1


def test_antiparallel_and_parallel_edges():
    # (0, 1) and (1, 0) join the same pair: the smallest weight is kept,
    # with its direction as given
    edges = pd.DataFrame(
        {
            "tail": [0, 1, 1, 1, 2, 2],
            "head": [1, 0, 2, 2, 1, 2],
            "weight": [5.0, 2.0, 7.0, 3.0, 4.0, 0.5],
        }
    )
    mst = MinimumSpanningTree(edges)
    forest = mst.run()
    assert list(zip(forest["tail"], forest["head"], forest["weight"])) == [
        (1, 0, 2.0),
        (1, 2, 3.0),
    ]
    assert mst.total_weight == 5.0
    assert mst.n_trees == 1


def test_forest():
    # two components, and vertices 3 and 4 that are not used by any edge
    edges = pd.DataFrame(
        {
            "tail": [0, 1, 2, 5, 6, 7],
            "head": [1, 2, 0, 6, 7, 5],
            "weight": [1.0, 2.0, 3.0, 1.0, 1.0, 1.0],
        }
    )
    mst = MinimumSpanningTree(edges)
    forest = mst.run()
    assert len(forest) == 4
    assert mst.total_weight == 5.0
    assert mst.n_trees == 2


def test_isolated_vertex_by_self_loop():
    edges = pd.DataFrame({"tail": [0, 2], "head": [1, 2], "weight": [1.0, 1.0]})
    mst = MinimumSpanningTree(edges)
    forest = mst.run()
    assert len(forest) == 1
    assert mst.n_trees == 2


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("n_vertices, n_edges", [(50, 60), (200, 800), (1000, 1500)])
def test_random_vs_scipy(seed, n_vertices, n_edges):
    edges = random_edges(n_vertices, n_edges, seed)
    mst = MinimumSpanningTree(edges)
    forest = mst.run()
    assert mst.total_weight == pytest.approx(scipy_mst_weight(edges, n_vertices))
    assert forest["weight"].sum() == pytest.approx(mst.total_weight)
    assert np.all(np.diff(forest["weight"].to_numpy()) >= 0.0)
    n_used = len(np.union1d(edges["tail"], edges["head"]))
    assert mst.n_trees == undirected_components(edges, n_vertices)
    assert len(forest) == n_used - mst.n_trees
    check_forest(edges, forest)


@pytest.mark.parametrize("seed", range(3))
def test_negative_weights(seed):
    # adding a constant to all the weights does not change the forest
    edges = random_edges(300, 900, seed, low=-5.0, high=5.0)
    shifted = edges.assign(weight=edges["weight"] + 10.0)
    mst = MinimumSpanningTree(edges)
    forest = mst.run()
    expected = scipy_mst_weight(shifted, 300) - 10.0 * len(forest)
    assert mst.total_weight == pytest.approx(expected)
    check_forest(edges, forest)


@pytest.mark.parametrize("seed", range(3))
def test_permute_negative_labels(seed):
    edges = random_edges(300, 600, seed)
    reference = MinimumSpanningTree(edges)
    reference_forest = reference.run()

    # sparse and negative labels
    shifted = edges.assign(tail=edges["tail"] * 7 - 1000, head=edges["head"] * 7 - 1000)
    mst = MinimumSpanningTree(shifted, permute=True)
    forest = mst.run()
    assert mst.total_weight == pytest.approx(reference.total_weight)
    assert mst.n_trees == reference.n_trees
    assert (forest["tail"] < 0).any()
    check_forest(shifted, forest)
    # same edges, since the renumbering keeps the order of the labels
    assert (forest["tail"].to_numpy() == reference_forest["tail"] * 7 - 1000).all()
    assert (forest["head"].to_numpy() == reference_forest["head"] * 7 - 1000).all()


def test_permute_with_gaps():
    edges = random_edges(500, 400, 3)
    a = MinimumSpanningTree(edges)
    b = MinimumSpanningTree(edges, permute=True)
    fa, fb = a.run(), b.run()
    assert a.n_trees == b.n_trees
    assert a.total_weight == pytest.approx(b.total_weight)
    pd.testing.assert_frame_equal(fa, fb)


def test_outputs():
    edges = random_edges(100, 300, 0)
    mst = MinimumSpanningTree(edges)
    forest = mst.run(output="pandas")
    tails, heads, weights = mst.run(output="numpy")
    assert isinstance(tails, np.ndarray)
    np.testing.assert_array_equal(tails, forest["tail"].to_numpy())
    np.testing.assert_array_equal(heads, forest["head"].to_numpy())
    np.testing.assert_array_equal(weights, forest["weight"].to_numpy())
    with pytest.raises(ValueError, match="output"):
        mst.run(output="arrow")


def test_polars():
    pl = pytest.importorskip("polars")
    edges = random_edges(100, 300, 1)
    expected = MinimumSpanningTree(edges).run()
    mst = MinimumSpanningTree(pl.from_pandas(edges))
    forest = mst.run(output="polars")
    assert isinstance(forest, pl.DataFrame)
    assert forest.columns == ["tail", "head", "weight"]
    np.testing.assert_array_equal(forest["tail"].to_numpy(), expected["tail"])
    np.testing.assert_array_equal(forest["weight"].to_numpy(), expected["weight"])


def test_column_names():
    edges = random_edges(50, 100, 2).rename(
        columns={"tail": "u", "head": "v", "weight": "cost"}
    )
    mst = MinimumSpanningTree(edges, tail="u", head="v", weight="cost")
    assert list(mst.run().columns) == ["u", "v", "cost"]


def test_unweighted():
    edges = random_edges(100, 150, 4)
    mst = MinimumSpanningTree(edges, weight=None)
    forest = mst.run()
    assert list(forest.columns) == ["tail", "head"]
    assert mst.total_weight == len(forest)
    assert mst.n_trees == undirected_components(edges, 100)


def test_ties_deterministic():
    edges = pd.DataFrame(
        {"tail": [2, 0, 1], "head": [0, 1, 2], "weight": [1.0, 1.0, 1.0]}
    )
    forest = MinimumSpanningTree(edges).run()
    assert list(zip(forest["tail"], forest["head"])) == [(0, 1), (1, 2)]


def test_empty_graph():
    edges = pd.DataFrame(
        {
            "tail": np.array([], dtype=np.int64),
            "head": np.array([], dtype=np.int64),
            "weight": np.array([], dtype=np.float64),
        }
    )
    mst = MinimumSpanningTree(edges)
    forest = mst.run()
    assert len(forest) == 0
    assert mst.total_weight == 0.0
    assert mst.n_trees == 0


def test_invalid_edges():
    edges = random_edges(10, 20, 0)
    with pytest.raises(KeyError):
        MinimumSpanningTree(edges, weight="cost", check_edges=True)
    bad = edges.copy()
    bad.loc[3, "weight"] = np.inf
    with pytest.raises(ValueError, match="finite"):
        MinimumSpanningTree(bad)
    bad.loc[3, "weight"] = np.nan
    with pytest.raises(ValueError):
        MinimumSpanningTree(bad, check_edges=True)
    with pytest.raises(ValueError):
        MinimumSpanningTree(bad)
    with pytest.raises(TypeError):
        MinimumSpanningTree(edges.to_numpy(), check_edges=True)


def test_kruskal_kernel():
    tails = np.array([0, 1, 2], dtype=np.uint32)
    heads = np.array([1, 2, 0], dtype=np.uint32)
    order = np.array([2, 0, 1], dtype=np.int64)
    np.testing.assert_array_equal(kruskal(tails, heads, order, 3), [2, 0])
    np.testing.assert_array_equal(kruskal(tails, heads, order[:1], 3), [2])


def test_kruskal_kernel_invalid():
    tails = np.array([0, 1, 2], dtype=np.uint32)
    heads = np.array([1, 2, 0], dtype=np.uint32)
    order = np.arange(3, dtype=np.int64)
    with pytest.raises(ValueError, match="same length"):
        kruskal(tails, heads[:2], order, 3)
    with pytest.raises(ValueError, match="vertex_count"):
        kruskal(tails, heads, order, 2)
    with pytest.raises(ValueError, match="non-negative"):
        kruskal(tails, heads, order, -1)
    with pytest.raises(ValueError, match="order"):
        kruskal(tails, heads, np.array([0, 3, 1], dtype=np.int64), 3)
    with pytest.raises(ValueError, match="order"):
        kruskal(tails, heads, np.array([0, -1, 1], dtype=np.int64), 3)
    with pytest.raises(ValueError, match="more than"):
        kruskal(tails, heads, np.arange(4, dtype=np.int64), 3)
