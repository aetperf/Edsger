"""Tests for the centrality measures (edsger.centrality, edsger.power_iteration)."""

import numpy as np
import pandas as pd
import pytest

from edsger.centrality import Centrality, ConvergenceError
from edsger.power_iteration import hits_csr_csc, katz_csc, pagerank_csc


def random_edges(n_vertices, n_edges, seed):
    """Random directed edges, with parallel edges, self-loops and dangling
    vertices, and positive weights."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "tail": rng.integers(0, n_vertices, n_edges),
            "head": rng.integers(0, n_vertices, n_edges),
            "weight": rng.uniform(0.5, 5.0, n_edges),
        }
    )


def dense_adjacency(edges, weighted):
    """Dense adjacency matrix over the vertices used by the edges, in the order of
    their labels, after the reduction of the parallel edges to their minimum
    weight (as in edsger)."""
    reduced = edges.groupby(["tail", "head"], as_index=False)["weight"].min()
    labels = np.union1d(reduced["tail"], reduced["head"])
    t = np.searchsorted(labels, reduced["tail"])
    h = np.searchsorted(labels, reduced["head"])
    a = np.zeros((len(labels), len(labels)))
    a[t, h] = reduced["weight"] if weighted else 1.0
    return labels, a


def pagerank_reference(a, damping=0.85, personalization=None):
    """PageRank by a linear solve: x = damping M^T x + (1 - damping) p, where M is
    the transition matrix with the rows of the dangling vertices replaced by p."""
    n = a.shape[0]
    p = np.full(n, 1.0 / n) if personalization is None else personalization
    p = p / p.sum()
    out = a.sum(axis=1)
    m = np.empty_like(a)
    dangling = out == 0.0
    m[~dangling] = a[~dangling] / out[~dangling, None]
    m[dangling] = p
    return np.linalg.solve(np.eye(n) - damping * m.T, (1.0 - damping) * p)


def hits_reference(a):
    """Principal singular vectors of the adjacency matrix, normalized to sum 1."""
    u, _, vt = np.linalg.svd(a)
    hubs, authorities = np.abs(u[:, 0]), np.abs(vt[0])
    return hubs / hubs.sum(), authorities / authorities.sum()


def katz_reference(a, alpha, beta=1.0, normalized=True):
    """Katz centrality by a linear solve: (I - alpha A^T) x = beta."""
    n = a.shape[0]
    x = np.linalg.solve(np.eye(n) - alpha * a.T, np.full(n, beta))
    return x / np.linalg.norm(x) if normalized else x


def safe_katz_alpha(a):
    return 0.5 / max(abs(np.linalg.eigvals(a)).max(), 1e-12)


def to_networkx(nx, edges, weighted):
    reduced = edges.groupby(["tail", "head"], as_index=False)["weight"].min()
    graph = nx.DiGraph()
    if weighted:
        graph.add_weighted_edges_from(
            zip(reduced["tail"], reduced["head"], reduced["weight"], strict=True)
        )
    else:
        graph.add_edges_from(zip(reduced["tail"], reduced["head"], strict=True))
    return graph


def labelled(result_series, labels):
    return result_series.loc[labels].to_numpy()


# ---------------------------------------------------------------- PageRank


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("seed", range(4))
def test_pagerank_vs_dense(weighted, seed):
    edges = random_edges(80, 200, seed)
    labels, a = dense_adjacency(edges, weighted)
    c = Centrality(edges, weight="weight" if weighted else None)
    result = c.pagerank(tol=1e-12, max_iter=1000, return_series=True)
    expected = pagerank_reference(a)
    np.testing.assert_allclose(labelled(result, labels), expected, atol=1e-9)
    assert result.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("damping", [0.0, 0.5, 0.99])
def test_pagerank_damping(damping):
    edges = random_edges(50, 150, 7)
    labels, a = dense_adjacency(edges, False)
    result = Centrality(edges).pagerank(
        damping=damping, tol=1e-13, max_iter=10000, output="pandas"
    )
    np.testing.assert_allclose(
        labelled(result, labels), pagerank_reference(a, damping), atol=1e-8
    )


def test_pagerank_personalization():
    edges = random_edges(60, 150, 3)
    labels, a = dense_adjacency(edges, True)
    rng = np.random.default_rng(0)
    chosen = rng.choice(labels, 10, replace=False)
    values = rng.uniform(0.0, 3.0, 10)
    p = np.zeros(len(labels))
    p[np.searchsorted(labels, chosen)] = values
    expected = pagerank_reference(a, 0.85, p)
    c = Centrality(edges, weight="weight")
    for personalization in (
        dict(zip(chosen.tolist(), values.tolist(), strict=True)),
        pd.Series(values, index=chosen),
    ):
        result = c.pagerank(
            personalization=personalization, tol=1e-12, max_iter=1000, output="pandas"
        )
        np.testing.assert_allclose(labelled(result, labels), expected, atol=1e-9)


def test_pagerank_dangling():
    # 3 is dangling (no out-edge), 2 has only an out-edge of weight 0: both
    # spread their value according to the personalization (uniform here)
    edges = pd.DataFrame(
        {"tail": [0, 1, 1, 2], "head": [1, 2, 3, 0], "weight": [1.0, 2.0, 1.0, 0.0]}
    )
    labels, a = dense_adjacency(edges, True)
    result = Centrality(edges, weight="weight").pagerank(tol=1e-12, max_iter=1000)
    np.testing.assert_allclose(result, pagerank_reference(a), atol=1e-10)
    assert result.sum() == pytest.approx(1.0)
    unweighted = Centrality(edges).pagerank(tol=1e-12, max_iter=1000)
    _, a1 = dense_adjacency(edges, False)
    np.testing.assert_allclose(unweighted, pagerank_reference(a1), atol=1e-10)


def test_pagerank_networkx():
    nx = pytest.importorskip("networkx")
    for weighted in (False, True):
        edges = random_edges(300, 1200, 11)
        graph = to_networkx(nx, edges, weighted)
        # NetworkX stops when the change is below n * tol: a tight tol makes
        # its result an accurate reference
        expected = nx.pagerank(
            graph, weight="weight" if weighted else None, tol=1e-13, max_iter=1000
        )
        c = Centrality(edges, weight="weight" if weighted else None)
        result = c.pagerank(output="pandas")
        nodes = np.array(list(expected))
        np.testing.assert_allclose(
            result.loc[nodes].to_numpy(), list(expected.values()), atol=1e-6
        )
        # personalization
        perso = {int(nodes[0]): 1.0, int(nodes[5]): 2.0}
        expected = nx.pagerank(
            graph,
            personalization=perso,
            weight="weight" if weighted else None,
            tol=1e-13,
            max_iter=1000,
        )
        result = c.pagerank(personalization=perso, output="pandas")
        np.testing.assert_allclose(
            result.loc[nodes].to_numpy(), list(expected.values()), atol=1e-6
        )


def test_pagerank_not_converged():
    edges = random_edges(50, 150, 0)
    with pytest.raises(ConvergenceError, match="PageRank did not converge"):
        Centrality(edges).pagerank(max_iter=2)


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"damping": 1.5}, ValueError),
        ({"damping": -0.1}, ValueError),
        ({"damping": np.nan}, ValueError),
        ({"damping": "0.8"}, TypeError),
        ({"tol": 0.0}, ValueError),
        ({"tol": -1.0}, ValueError),
        ({"max_iter": 0}, ValueError),
        ({"max_iter": 1.5}, TypeError),
        ({"personalization": {0: -1.0}}, ValueError),
        ({"personalization": {0: 0.0}}, ValueError),
        ({"personalization": {12345: 1.0}}, ValueError),
        ({"personalization": {0.5: 1.0}}, TypeError),
        ({"personalization": [0.1, 0.2]}, TypeError),
        ({"personalization": {0: np.inf}}, ValueError),
        ({"output": "arrow"}, ValueError),
    ],
)
def test_pagerank_invalid(kwargs, error):
    edges = random_edges(20, 50, 0)
    with pytest.raises(error):
        Centrality(edges).pagerank(**kwargs)


# ---------------------------------------------------------------- HITS


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("seed", range(4))
def test_hits_vs_dense(weighted, seed):
    edges = random_edges(80, 300, seed)
    labels, a = dense_adjacency(edges, weighted)
    c = Centrality(edges, weight="weight" if weighted else None)
    hubs, authorities = c.hits(tol=1e-12, max_iter=10000, output="pandas")
    expected_hubs, expected_authorities = hits_reference(a)
    np.testing.assert_allclose(labelled(hubs, labels), expected_hubs, atol=1e-8)
    np.testing.assert_allclose(
        labelled(authorities, labels), expected_authorities, atol=1e-8
    )
    assert hubs.name == "hub" and authorities.name == "authority"
    assert hubs.sum() == pytest.approx(1.0)
    assert authorities.sum() == pytest.approx(1.0)


def test_hits_networkx():
    nx = pytest.importorskip("networkx")
    for weighted in (False, True):
        edges = random_edges(300, 1500, 5)
        graph = to_networkx(nx, edges, weighted)
        if weighted:
            expected_h, expected_a = nx.hits(graph, max_iter=1000, tol=1e-10)
        else:
            unweighted = nx.DiGraph(graph.edges())
            expected_h, expected_a = nx.hits(unweighted, max_iter=1000, tol=1e-10)
        c = Centrality(edges, weight="weight" if weighted else None)
        hubs, authorities = c.hits(tol=1e-10, max_iter=1000, output="pandas")
        nodes = np.array(list(expected_h))
        np.testing.assert_allclose(
            hubs.loc[nodes].to_numpy(), list(expected_h.values()), atol=1e-6
        )
        np.testing.assert_allclose(
            authorities.loc[nodes].to_numpy(),
            [expected_a[v] for v in nodes],
            atol=1e-6,
        )


def test_hits_not_converged():
    edges = random_edges(50, 150, 0)
    with pytest.raises(ConvergenceError, match="HITS did not converge"):
        Centrality(edges).hits(max_iter=1)


def test_hits_zero_weights():
    edges = pd.DataFrame({"tail": [0, 1], "head": [1, 2], "weight": [0.0, 0.0]})
    with pytest.raises(ValueError, match="all the edge weights are 0"):
        Centrality(edges, weight="weight").hits()


# ---------------------------------------------------------------- Katz


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("seed", range(4))
def test_katz_vs_dense(weighted, seed):
    edges = random_edges(80, 250, seed)
    labels, a = dense_adjacency(edges, weighted)
    alpha = safe_katz_alpha(a)
    c = Centrality(edges, weight="weight" if weighted else None)
    result = c.katz(alpha=alpha, beta=2.0, tol=1e-13, output="pandas")
    np.testing.assert_allclose(
        labelled(result, labels), katz_reference(a, alpha, 2.0), atol=1e-10
    )
    raw = c.katz(alpha=alpha, beta=2.0, tol=1e-13, normalized=False, output="pandas")
    np.testing.assert_allclose(
        labelled(raw, labels), katz_reference(a, alpha, 2.0, False), rtol=1e-9
    )


def test_katz_networkx():
    nx = pytest.importorskip("networkx")
    for weighted in (False, True):
        edges = random_edges(300, 900, 2)
        graph = to_networkx(nx, edges, weighted)
        _, a = dense_adjacency(edges, weighted)
        alpha = safe_katz_alpha(a)
        expected = nx.katz_centrality(
            graph, alpha=alpha, weight="weight" if weighted else None
        )
        c = Centrality(edges, weight="weight" if weighted else None)
        result = c.katz(alpha=alpha, output="pandas")
        nodes = np.array(list(expected))
        np.testing.assert_allclose(
            result.loc[nodes].to_numpy(), list(expected.values()), atol=1e-6
        )


def test_katz_diverging():
    edges = random_edges(50, 300, 0)
    _, a = dense_adjacency(edges, False)
    alpha = 2.0 / abs(np.linalg.eigvals(a)).max()
    with pytest.raises(ConvergenceError, match="spectral radius"):
        Centrality(edges).katz(alpha=alpha)
    with pytest.raises(ConvergenceError):
        Centrality(edges).katz(alpha=safe_katz_alpha(a), max_iter=2)


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"alpha": np.inf}, ValueError),
        ({"alpha": None}, TypeError),
        ({"beta": np.nan}, ValueError),
        ({"beta": "1"}, TypeError),
        ({"tol": 0.0}, ValueError),
        ({"max_iter": -3}, ValueError),
    ],
)
def test_katz_invalid(kwargs, error):
    edges = random_edges(20, 50, 0)
    with pytest.raises(error):
        Centrality(edges).katz(**kwargs)


def test_katz_beta_zero():
    edges = random_edges(20, 50, 0)
    result = Centrality(edges).katz(beta=0.0)
    assert (result == 0.0).all()


# ---------------------------------------------------------------- graph and outputs


def test_vertices_with_gaps_and_permute():
    # vertex identifiers 0, 10, 20, ...: the other integers are not vertices
    edges = random_edges(40, 120, 9)
    sparse = edges.assign(tail=edges["tail"] * 10, head=edges["head"] * 10)
    labels, a = dense_adjacency(sparse, True)
    for permute in (False, True):
        c = Centrality(sparse, weight="weight", permute=permute)
        pr = c.pagerank(tol=1e-12, max_iter=1000)
        assert len(pr) == labels.max() + 1
        not_vertex = np.ones(len(pr), dtype=bool)
        not_vertex[labels] = False
        assert (pr[not_vertex] == 0.0).all()
        np.testing.assert_allclose(pr[labels], pagerank_reference(a), atol=1e-9)
        hubs, authorities = c.hits(tol=1e-12, max_iter=10000)
        expected_hubs, expected_authorities = hits_reference(a)
        np.testing.assert_allclose(hubs[labels], expected_hubs, atol=1e-8)
        np.testing.assert_allclose(authorities[labels], expected_authorities, atol=1e-8)
        alpha = safe_katz_alpha(a)
        katz = c.katz(alpha=alpha, tol=1e-13)
        assert (katz[not_vertex] == 0.0).all()
        np.testing.assert_allclose(katz[labels], katz_reference(a, alpha), atol=1e-10)


def test_negative_labels():
    edges = random_edges(40, 120, 1)
    shifted = edges.assign(tail=edges["tail"] - 20, head=edges["head"] - 20)
    expected = Centrality(edges).pagerank(output="pandas")
    c = Centrality(shifted, permute=True)
    result = c.pagerank(output="pandas")
    np.testing.assert_allclose(result.to_numpy(), expected.loc[expected > 0])
    assert (result.index == expected.index[expected > 0] - 20).all()
    with pytest.raises(ValueError, match="negative vertex labels"):
        c.pagerank()
    with pytest.raises(ValueError, match="negative vertex labels"):
        c.hits(output="numpy")


def test_outputs():
    edges = random_edges(30, 90, 4)
    c = Centrality(edges)
    array = c.pagerank()
    assert isinstance(array, np.ndarray)
    series = c.pagerank(return_series=True)
    assert series.name == "pagerank" and series.index.name == "vertex_idx"
    np.testing.assert_array_equal(series.to_numpy(), array)
    katz = c.katz(output="pandas")
    assert katz.name == "katz"
    with pytest.raises(ValueError):
        c.katz(output="numpy", return_series=True)


def test_polars_output():
    pl = pytest.importorskip("polars")
    edges = random_edges(30, 90, 4)
    c = Centrality(pl.from_pandas(edges), weight="weight")
    expected = Centrality(edges, weight="weight").pagerank(output="pandas")
    result = c.pagerank(output="polars")
    assert isinstance(result, pl.DataFrame)
    assert result.columns == ["vertex_idx", "pagerank"]
    np.testing.assert_allclose(result["pagerank"].to_numpy(), expected.to_numpy())
    hubs, authorities = c.hits(output="polars")
    assert hubs.columns == ["vertex_idx", "hub"]
    assert authorities.columns == ["vertex_idx", "authority"]


def test_parallel_edges_min_weight():
    # the parallel edges are reduced to their minimum weight
    edges = pd.DataFrame(
        {"tail": [0, 0, 0, 1], "head": [1, 1, 2, 0], "weight": [5.0, 1.0, 2.0, 1.0]}
    )
    reduced = pd.DataFrame(
        {"tail": [0, 0, 1], "head": [1, 2, 0], "weight": [1.0, 2.0, 1.0]}
    )
    np.testing.assert_allclose(
        Centrality(edges, weight="weight").pagerank(),
        Centrality(reduced, weight="weight").pagerank(),
    )


def test_empty_graph():
    edges = pd.DataFrame(
        {"tail": np.array([], dtype=np.int64), "head": np.array([], dtype=np.int64)}
    )
    c = Centrality(edges)
    assert len(c.pagerank()) == 0
    assert len(c.katz()) == 0
    hubs, authorities = c.hits()
    assert len(hubs) == 0 and len(authorities) == 0


def test_invalid_weights():
    edges = random_edges(20, 50, 0)
    for value in (-1.0, np.nan, np.inf):
        bad = edges.copy()
        bad.loc[2, "weight"] = value
        with pytest.raises(ValueError):
            Centrality(bad, weight="weight")
    with pytest.raises(ValueError):
        Centrality(bad, weight="weight", check_edges=True)
    with pytest.raises(KeyError):
        Centrality(edges, weight="cost", check_edges=True)


# ---------------------------------------------------------------- kernels


def small_csc():
    indptr = np.array([0, 1, 2, 3], dtype=np.uint32)
    indices = np.array([2, 0, 1], dtype=np.uint32)
    return indptr, indices, np.ones(3)


def test_pagerank_kernel_invalid():
    indptr, indices, data = small_csc()
    v = np.full(3, 1.0 / 3.0)
    args = [indptr, indices, data, True, np.ones(3), v, v, v, 0.85, 1e-9, 100]
    x, n_iter = pagerank_csc(*args)
    np.testing.assert_allclose(x, v)
    assert n_iter > 0

    def call(position, value):
        bad = list(args)
        bad[position] = value
        pagerank_csc(*bad)

    with pytest.raises(ValueError, match="pointers"):
        call(0, indptr[:3])
    with pytest.raises(ValueError, match="end at the edge count"):
        call(1, indices[:2])
    with pytest.raises(ValueError, match="smaller than the vertex count"):
        call(1, np.array([2, 0, 3], dtype=np.uint32))
    with pytest.raises(ValueError, match="non-decreasing"):
        call(0, np.array([0, 2, 1, 3], dtype=np.uint32))
    with pytest.raises(ValueError, match="edge data"):
        call(2, data[:2])
    with pytest.raises(ValueError, match="teleport"):
        call(5, v[:2])
    with pytest.raises(ValueError, match="damping"):
        call(8, 1.5)
    with pytest.raises(ValueError, match="tol"):
        call(9, 0.0)
    with pytest.raises(ValueError, match="max_iter"):
        call(10, 0)
    # unweighted: the data array is not used
    unweighted = list(args)
    unweighted[2:4] = [np.empty(0), False]
    pagerank_csc(*unweighted)


def test_hits_katz_kernels_invalid():
    indptr, indices, data = small_csc()
    v = np.full(3, 1.0 / 3.0)
    h, a, n_iter = hits_csr_csc(
        indptr, indices, data, indptr, indices, data, True, v, 1e-9, 100
    )
    assert n_iter > 0
    with pytest.raises(ValueError, match="same edge count"):
        hits_csr_csc(
            indptr,
            indices,
            data,
            np.array([0, 1, 1, 2], dtype=np.uint32),
            indices[:2],
            data[:2],
            True,
            v,
            1e-9,
            100,
        )
    with pytest.raises(ValueError, match="h0"):
        hits_csr_csc(indptr, indices, data, indptr, indices, data, True, v[:1], 1, 1)
    x, n_iter = katz_csc(indptr, indices, data, True, np.ones(3), 0.1, 1e-9, 100)
    np.testing.assert_allclose(x, np.full(3, 1.0 / 0.9))
    with pytest.raises(ValueError, match="beta"):
        katz_csc(indptr, indices, data, True, np.ones(4), 0.1, 1e-9, 100)
    with pytest.raises(ValueError, match="pointer array"):
        katz_csc(indptr[:1], indices, data, True, np.ones(0), 0.1, 1e-9, 100)
