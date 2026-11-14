"""Tests of the package interface: top-level imports, representations, typing."""

import pathlib
import re

import pandas as pd
import pytest

import edsger
from edsger import path


def test_top_level_imports():
    assert edsger.Dijkstra is path.Dijkstra
    assert edsger.AStar is path.AStar
    for name in edsger.__all__:
        assert hasattr(edsger, name), name


def test_typed_package():
    package_dir = pathlib.Path(edsger.__file__).parent
    assert (package_dir / "py.typed").exists()
    # every extension module built by setup.py has a stub
    setup_py = pathlib.Path(__file__).parents[1] / "setup.py"
    if not setup_py.exists():
        pytest.skip("setup.py not available")
    modules = re.findall(r'"edsger\.(\w+)"', setup_py.read_text())
    assert "dijkstra" in modules
    for module in modules:
        assert (package_dir / f"{module}.pyi").exists(), module


@pytest.fixture
def edges():
    return pd.DataFrame(
        {
            "tail": [0, 0, 1],
            "head": [1, 2, 2],
            "weight": [1.0, 2.0, 1.0],
            "trav_time": [1.0, 2.0, 1.0],
            "freq": [1.0, 1.0, 1.0],
        }
    )


def test_repr(edges):
    assert repr(edsger.Dijkstra(edges)) == (
        "Dijkstra(n_vertices=3, n_edges=3, permute=False, orientation='out', lazy=True)"
    )
    assert repr(edsger.BFS(edges, orientation="in", permute=True)) == (
        "BFS(n_vertices=3, n_edges=3, permute=True, orientation='in')"
    )
    assert repr(edsger.AStar(edges)) == "AStar(n_vertices=3, n_edges=3, permute=False)"
    assert repr(edsger.Centrality(edges, weight="weight")).endswith("weighted=True)")
    assert repr(edsger.HyperpathGenerating(edges)) == (
        "HyperpathGenerating(vertex_count=3, edge_count=3, orientation='in')"
    )


def test_repr_large_numbers_are_readable():
    edges = pd.DataFrame({"tail": [0], "head": [1_234_566], "weight": [1.0]})
    assert "n_vertices=1_234_567" in repr(edsger.Dijkstra(edges))
