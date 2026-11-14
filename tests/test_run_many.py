"""Tests of Dijkstra.run_many: several runs in parallel."""

import numpy as np
import pandas as pd
import pytest

from edsger.path import Dijkstra


def random_edges(seed=0, n=200, m=1000, scale=1, offset=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "tail": rng.integers(0, n, m) * scale + offset,
            "head": rng.integers(0, n, m) * scale + offset,
            "weight": rng.random(m),
        }
    )


@pytest.mark.parametrize("orientation", ["out", "in"])
@pytest.mark.parametrize("permute", [False, True])
@pytest.mark.parametrize("n_jobs", [1, 4])
def test_run_many_same_as_run(orientation, permute, n_jobs):
    edges = random_edges(scale=3)
    sp = Dijkstra(edges, orientation=orientation, permute=permute)
    vertices = sp.get_vertices()[:20]
    result = sp.run_many(vertices, n_jobs=n_jobs)
    expected = np.vstack([sp.run(int(v)) for v in vertices])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("return_inf", [True, False])
def test_run_many_return_inf(return_inf):
    edges = pd.DataFrame({"tail": [0, 1], "head": [1, 2], "weight": [1.0, 1.0]})
    sp = Dijkstra(edges)
    result = sp.run_many([2, 0], return_inf=return_inf)
    np.testing.assert_array_equal(result[1], sp.run(0, return_inf=return_inf))
    np.testing.assert_array_equal(result[0], sp.run(2, return_inf=return_inf))


def test_run_many_termination_nodes():
    edges = random_edges(seed=1)
    sp = Dijkstra(edges)
    origins = [0, 5, 7]
    destinations = [3, 1, 150]
    result = sp.run_many(origins, termination_nodes=destinations)
    assert result.shape == (3, 3)
    full = sp.run_many(origins)
    np.testing.assert_array_equal(result, full[:, destinations])


def test_run_many_pandas_output():
    edges = random_edges(seed=2, scale=10)
    sp = Dijkstra(edges, permute=True)
    vertices = sp.get_vertices()[:5]
    result = sp.run_many(vertices, output="pandas")
    assert isinstance(result, pd.DataFrame)
    assert result.index.name == "origin"
    assert result.columns.name == "destination"
    np.testing.assert_array_equal(result.index, vertices)
    np.testing.assert_array_equal(result.columns, sp.get_vertices())
    for v in vertices:
        pd.testing.assert_series_equal(
            result.loc[v],
            sp.run(int(v), output="pandas"),
            check_names=False,
        )


def test_run_many_pandas_output_in_orientation():
    edges = random_edges(seed=2)
    result = Dijkstra(edges, orientation="in").run_many([0, 1], output="pandas")
    assert result.index.name == "destination"
    assert result.columns.name == "origin"


def test_run_many_polars_output():
    pl = pytest.importorskip("polars")
    edges = random_edges(seed=3)
    sp = Dijkstra(edges)
    result = sp.run_many([4, 2], termination_nodes=[0, 1, 9], output="polars")
    assert isinstance(result, pl.DataFrame)
    assert result.columns == ["origin", "destination", "path_length"]
    assert result["origin"].to_list() == [4, 4, 4, 2, 2, 2]
    assert result["destination"].to_list() == [0, 1, 9, 0, 1, 9]
    matrix = sp.run_many([4, 2], termination_nodes=[0, 1, 9])
    np.testing.assert_array_equal(result["path_length"].to_numpy(), matrix.ravel())


def test_run_many_polars_output_in_orientation():
    pl = pytest.importorskip("polars")
    edges = random_edges(seed=3)
    sp = Dijkstra(edges, orientation="in")
    result = sp.run_many([4], termination_nodes=[0, 1], output="polars")
    assert isinstance(result, pl.DataFrame)
    # the run vertex is the destination
    assert result["destination"].to_list() == [4, 4]
    assert result["origin"].to_list() == [0, 1]


def test_run_many_negative_labels():
    edges = random_edges(seed=4, offset=-50)
    sp = Dijkstra(edges, permute=True)
    with pytest.raises(ValueError, match="negative vertex labels"):
        sp.run_many([-50])
    result = sp.run_many([-50, -49], output="pandas")
    pd.testing.assert_series_equal(
        result.loc[-49], sp.run(-49, output="pandas"), check_names=False
    )


def test_run_many_invalid_arguments():
    sp = Dijkstra(random_edges())
    with pytest.raises(ValueError, match="invalid vertex indices"):
        sp.run_many([0, 10_000])
    with pytest.raises(ValueError, match="1-dimensional"):
        sp.run_many([[0, 1]])
    with pytest.raises(TypeError, match="n_jobs"):
        sp.run_many([0], n_jobs=1.5)
    with pytest.raises(ValueError, match="n_jobs"):
        sp.run_many([0], n_jobs=0)
    sp_permuted = Dijkstra(random_edges(scale=3), permute=True)
    with pytest.raises(ValueError, match="vertex 1 not found"):
        sp_permuted.run_many([0, 1])


def test_run_many_no_vertices():
    sp = Dijkstra(random_edges())
    assert sp.run_many([]).shape == (0, sp.n_vertices)
