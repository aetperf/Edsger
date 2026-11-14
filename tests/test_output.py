"""Tests of the `output` argument: NumPy, pandas or Polars results."""

import numpy as np
import pandas as pd
import pytest

from edsger.path import BFS, BellmanFord, Dijkstra

pl = pytest.importorskip("polars")


@pytest.fixture
def edges():
    return pd.DataFrame(
        {
            "tail": [0, 0, 10, 10, 20],
            "head": [10, 20, 20, 30, 30],
            "weight": [1.0, 2.0, 0.0, 2.0, 1.0],
        }
    )


@pytest.mark.parametrize("algo", [Dijkstra, BellmanFord])
@pytest.mark.parametrize("permute", [False, True])
def test_polars_output_path_lengths(edges, algo, permute):
    sp = algo(edges, permute=permute)
    expected = sp.run(vertex_idx=0, output="pandas")
    result = sp.run(vertex_idx=0, output="polars")
    assert isinstance(result, pl.DataFrame)
    assert result.columns == ["vertex_idx", "path_length"]
    assert result["path_length"].dtype == pl.Float64
    np.testing.assert_array_equal(result["vertex_idx"].to_numpy(), expected.index)
    np.testing.assert_array_equal(result["path_length"].to_numpy(), expected.values)


def test_polars_output_one_row_per_vertex_with_permute(edges):
    result = Dijkstra(edges, permute=True).run(vertex_idx=0, output="polars")
    assert result["vertex_idx"].to_list() == [0, 10, 20, 30]
    assert result["path_length"].to_list() == [0.0, 1.0, 1.0, 2.0]


def test_polars_output_bfs(edges):
    bfs = BFS(edges, permute=True)
    result = bfs.run(vertex_idx=0, output="polars")
    assert result.columns == ["vertex_idx", "predecessor"]
    assert result["predecessor"].dtype == pl.Int32
    assert result["predecessor"].to_list() == [bfs.UNREACHABLE, 0, 0, 10]


def test_polars_output_termination_nodes(edges):
    result = Dijkstra(edges).run(
        vertex_idx=0, termination_nodes=[30, 10], output="polars"
    )
    assert result["vertex_idx"].to_list() == [30, 10]
    assert result["path_length"].to_list() == [2.0, 1.0]


@pytest.mark.parametrize("algo", [Dijkstra, BellmanFord, BFS])
def test_negative_labels_polars_output(algo):
    edges = pd.DataFrame(
        {"tail": [-5, -5, 3], "head": [3, 7, 7], "weight": [1.0, 4.0, 1.0]}
    )
    sp = algo(edges, permute=True)
    result = sp.run(vertex_idx=-5, output="polars")
    assert result["vertex_idx"].to_list() == [-5, 3, 7]
    with pytest.raises(ValueError, match="output='pandas'"):
        sp.run(vertex_idx=-5)


@pytest.mark.parametrize("algo", [Dijkstra, BellmanFord, BFS])
def test_output_pandas_is_return_series(edges, algo):
    sp = algo(edges, permute=True)
    pd.testing.assert_series_equal(
        sp.run(vertex_idx=0, output="pandas"), sp.run(vertex_idx=0, return_series=True)
    )


@pytest.mark.parametrize("algo", [Dijkstra, BellmanFord])
def test_series_naming_is_the_same_for_all_algorithms(edges, algo):
    for permute in [False, True]:
        result = algo(edges, permute=permute).run(vertex_idx=0, output="pandas")
        assert result.name == "path_length"
        assert result.index.name == "vertex_idx"


@pytest.mark.parametrize("algo", [Dijkstra, BellmanFord, BFS])
def test_output_numpy_is_the_default(edges, algo):
    sp = algo(edges)
    np.testing.assert_array_equal(
        sp.run(vertex_idx=0), sp.run(vertex_idx=0, output="numpy")
    )


def test_polars_input_numpy_output_by_default(edges):
    result = Dijkstra(pl.from_pandas(edges)).run(vertex_idx=0)
    assert isinstance(result, np.ndarray)


def test_invalid_output(edges):
    sp = Dijkstra(edges)
    with pytest.raises(ValueError, match="argument 'output'"):
        sp.run(vertex_idx=0, output="arrow")
    with pytest.raises(ValueError, match="conflicts"):
        sp.run(vertex_idx=0, return_series=True, output="polars")


@pytest.mark.parametrize("algo", [Dijkstra, BellmanFord, BFS])
def test_get_path_after_polars_output(edges, algo):
    sp = algo(edges, permute=True)
    sp.run(vertex_idx=0, path_tracking=True)
    expected = sp.get_path(30)
    sp.run(vertex_idx=0, path_tracking=True, output="polars")
    np.testing.assert_array_equal(sp.get_path(30), expected)
    assert expected[0] == 30 and expected[-1] == 0
