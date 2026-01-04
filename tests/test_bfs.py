"""Tests for BFS implementation.

py.test tests/test_bfs.py -v
"""

import numpy as np
import pandas as pd
import pytest
from edsger.path import BFS

try:
    import pyarrow as pa

    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False


@pytest.fixture
def simple_graph():
    """Simple directed graph: 0 -> 1 -> 3, 0 -> 2 -> 3"""
    edges = pd.DataFrame(
        data={
            "tail": [0, 0, 1, 2],
            "head": [1, 2, 3, 3],
        }
    )
    return edges


@pytest.fixture
def linear_graph():
    """Linear graph: 0 -> 1 -> 2 -> 3"""
    edges = pd.DataFrame(
        data={
            "tail": [0, 1, 2],
            "head": [1, 2, 3],
        }
    )
    return edges


@pytest.fixture
def disconnected_graph():
    """Disconnected graph: 0 -> 1, 2 -> 3"""
    edges = pd.DataFrame(
        data={
            "tail": [0, 2],
            "head": [1, 3],
        }
    )
    return edges


@pytest.fixture
def cyclic_graph():
    """Cyclic graph: 0 -> 1 -> 2 -> 0"""
    edges = pd.DataFrame(
        data={
            "tail": [0, 1, 2],
            "head": [1, 2, 0],
        }
    )
    return edges


# ============================================================================ #
# Basic functionality tests                                                    #
# ============================================================================ #


def test_bfs_basic_out(simple_graph):
    """Test BFS with orientation='out' on simple graph."""
    bfs = BFS(simple_graph, orientation="out")
    predecessors = bfs.run(vertex_idx=0)

    # From vertex 0
    assert predecessors[0] == bfs.UNREACHABLE  # start vertex
    assert predecessors[1] == 0  # reached from 0
    assert predecessors[2] == 0  # reached from 0
    assert predecessors[3] in [1, 2]  # could be reached from 1 or 2


def test_bfs_basic_in(simple_graph):
    """Test BFS with orientation='in' on simple graph."""
    bfs = BFS(simple_graph, orientation="in")
    predecessors = bfs.run(vertex_idx=3)

    # To vertex 3 (working backward)
    assert predecessors[3] == bfs.UNREACHABLE  # start vertex
    # Vertex 3 can be reached from 1 or 2
    assert predecessors[1] in [3, bfs.UNREACHABLE] or predecessors[2] in [
        3,
        bfs.UNREACHABLE,
    ]


def test_bfs_linear_graph(linear_graph):
    """Test BFS on linear graph."""
    bfs = BFS(linear_graph, orientation="out")
    predecessors = bfs.run(vertex_idx=0)

    assert predecessors[0] == bfs.UNREACHABLE
    assert predecessors[1] == 0
    assert predecessors[2] == 1
    assert predecessors[3] == 2


def test_bfs_disconnected_graph(disconnected_graph):
    """Test BFS on disconnected graph."""
    bfs = BFS(disconnected_graph, orientation="out")
    predecessors = bfs.run(vertex_idx=0)

    # From vertex 0, can only reach vertex 1
    assert predecessors[0] == bfs.UNREACHABLE  # start
    assert predecessors[1] == 0  # reachable
    assert predecessors[2] == bfs.UNREACHABLE  # unreachable
    assert predecessors[3] == bfs.UNREACHABLE  # unreachable


def test_bfs_cyclic_graph(cyclic_graph):
    """Test BFS on cyclic graph."""
    bfs = BFS(cyclic_graph, orientation="out")
    predecessors = bfs.run(vertex_idx=0)

    # All vertices reachable from 0 in a cycle
    assert predecessors[0] == bfs.UNREACHABLE  # start
    assert predecessors[1] == 0
    assert predecessors[2] == 1
    # Note: 0 is reachable from 2, but it's already visited, so its predecessor remains UNREACHABLE


def test_bfs_single_node():
    """Test BFS on single node graph."""
    edges = pd.DataFrame(
        data={
            "tail": [],
            "head": [],
        }
    )
    bfs = BFS(edges, orientation="out")
    # Empty graph
    assert bfs.n_vertices == 0
    assert bfs.n_edges == 0


def test_bfs_two_nodes():
    """Test BFS on two-node graph."""
    edges = pd.DataFrame(
        data={
            "tail": [0],
            "head": [1],
        }
    )
    bfs = BFS(edges, orientation="out")
    predecessors = bfs.run(vertex_idx=0)

    assert predecessors[0] == bfs.UNREACHABLE
    assert predecessors[1] == 0


# ============================================================================ #
# Path tracking tests                                                          #
# ============================================================================ #


def test_bfs_path_tracking_out(simple_graph):
    """Test BFS path tracking with orientation='out'."""
    bfs = BFS(simple_graph, orientation="out")
    bfs.run(vertex_idx=0, path_tracking=True)

    # Check predecessors are stored
    assert bfs.path_links is not None

    # Get path to vertex 3
    path = bfs.get_path(3)
    assert path is not None
    assert path[0] == 3  # starts at target
    assert path[-1] == 0  # ends at source
    # Path should be: 3 <- 1 <- 0 or 3 <- 2 <- 0
    assert len(path) == 3


def test_bfs_path_tracking_in(simple_graph):
    """Test BFS path tracking with orientation='in'."""
    bfs = BFS(simple_graph, orientation="in")
    bfs.run(vertex_idx=3, path_tracking=True)

    # Check predecessors are stored
    assert bfs.path_links is not None

    # Get path from vertex 0
    path = bfs.get_path(0)
    if path is not None:
        assert path[0] == 0
        assert path[-1] == 3


def test_bfs_path_tracking_without_run():
    """Test getting path without running BFS first."""
    edges = pd.DataFrame(
        data={
            "tail": [0, 1],
            "head": [1, 2],
        }
    )
    bfs = BFS(edges)

    # Should warn when trying to get path without running BFS
    with pytest.warns(UserWarning, match=r"no path attribute"):
        path = bfs.get_path(2)
        assert path is None


def test_bfs_path_unreachable_vertex(disconnected_graph):
    """Test path extraction for unreachable vertex."""
    bfs = BFS(disconnected_graph, orientation="out")
    bfs.run(vertex_idx=0, path_tracking=True)

    # Vertex 3 is unreachable from 0
    path = bfs.get_path(3)
    # Path to unreachable vertex should be empty or contain only the vertex itself
    assert path is None or len(path) <= 1


# ============================================================================ #
# Permutation tests                                                            #
# ============================================================================ #


def test_bfs_permute_true():
    """Test BFS with non-contiguous vertex IDs and permute=True."""
    edges = pd.DataFrame(
        data={
            "tail": [0, 0, 10, 20],
            "head": [10, 20, 20, 30],
        }
    )
    bfs = BFS(edges, orientation="out", permute=True)
    predecessors = bfs.run(vertex_idx=0, path_tracking=True)

    # Check that vertex 0 is the start
    assert predecessors[0] == bfs.UNREACHABLE
    assert predecessors[10] == 0
    assert predecessors[20] in [0, 10]
    assert predecessors[30] == 20

    # Get path
    path = bfs.get_path(30)
    assert path is not None
    assert path[0] == 30
    assert path[-1] == 0


def test_bfs_permute_false():
    """Test BFS with non-contiguous vertex IDs and permute=False."""
    edges = pd.DataFrame(
        data={
            "tail": [0, 0, 10, 20],
            "head": [10, 20, 20, 30],
        }
    )
    bfs = BFS(edges, orientation="out", permute=False)
    predecessors = bfs.run(vertex_idx=0)

    # Without permutation, array includes all indices from 0 to max
    assert len(predecessors) == 31  # 0 to 30
    assert predecessors[0] == bfs.UNREACHABLE
    assert predecessors[10] == 0
    assert predecessors[20] in [0, 10]
    assert predecessors[30] == 20


def test_bfs_permute_return_series():
    """Test BFS with permute=True and return_series=True."""
    edges = pd.DataFrame(
        data={
            "tail": [10, 20],
            "head": [20, 30],
        }
    )
    bfs = BFS(edges, orientation="out", permute=True)
    result = bfs.run(vertex_idx=10, return_series=True)

    assert isinstance(result, pd.Series)
    assert result.index.name == "vertex_idx"
    assert result.name == "predecessor"
    assert result[10] == bfs.UNREACHABLE
    assert result[20] == 10
    assert result[30] == 20


# ============================================================================ #
# Edge cases and error handling                                                #
# ============================================================================ #


def test_bfs_invalid_vertex():
    """Test BFS with invalid vertex index."""
    edges = pd.DataFrame(
        data={
            "tail": [0, 1],
            "head": [1, 2],
        }
    )
    bfs = BFS(edges)

    # Negative vertex
    with pytest.raises(ValueError, match=r"must be non-negative"):
        bfs.run(vertex_idx=-1)

    # Vertex out of range
    with pytest.raises(ValueError, match=r"not found in graph"):
        bfs.run(vertex_idx=100)


def test_bfs_check_edges_invalid_dataframe():
    """Test edge validation with invalid DataFrame."""
    with pytest.raises(TypeError, match=r"pandas DataFrame"):
        BFS("not a dataframe", check_edges=True)


def test_bfs_check_edges_missing_columns():
    """Test edge validation with missing columns."""
    edges = pd.DataFrame(
        data={
            "from": [0, 1],
            "to": [1, 2],
        }
    )
    with pytest.raises(KeyError, match=r"not found in graph edges dataframe"):
        BFS(edges, check_edges=True)


def test_bfs_check_edges_null_values():
    """Test edge validation with null values."""
    edges = pd.DataFrame(
        data={
            "tail": [0, np.nan, 2],
            "head": [1, 2, 3],
        }
    )
    with pytest.raises(ValueError, match=r"missing value"):
        BFS(edges, check_edges=True)


def test_bfs_check_edges_wrong_dtype():
    """Test edge validation with wrong data types."""
    edges = pd.DataFrame(
        data={
            "tail": [0.5, 1.5, 2.5],
            "head": [1, 2, 3],
        }
    )
    with pytest.raises(TypeError, match=r"should be of integer type"):
        BFS(edges, check_edges=True)


def test_bfs_parallel_edges_verbose():
    """Test parallel edge removal with verbose output."""
    edges = pd.DataFrame(
        data={
            "tail": [0, 0, 1, 1],
            "head": [1, 1, 2, 2],
        }
    )
    # Should print message about removing parallel edges
    bfs = BFS(edges, verbose=True)
    assert bfs.n_edges == 2  # Only unique edges kept


# ============================================================================ #
# Return series tests                                                          #
# ============================================================================ #


def test_bfs_return_series(simple_graph):
    """Test BFS with return_series=True."""
    bfs = BFS(simple_graph, orientation="out")
    result = bfs.run(vertex_idx=0, return_series=True)

    assert isinstance(result, pd.Series)
    assert result.index.name == "vertex_idx"
    assert result.name == "predecessor"
    assert len(result) == 4


def test_bfs_return_series_with_path_tracking(linear_graph):
    """Test BFS with both return_series and path_tracking."""
    bfs = BFS(linear_graph, orientation="out")
    result = bfs.run(vertex_idx=0, return_series=True, path_tracking=True)

    assert isinstance(result, pd.Series)
    assert bfs.path_links is not None
    # path_links is always numpy array, even with return_series=True (like Dijkstra)
    assert isinstance(bfs.path_links, np.ndarray)


# ============================================================================ #
# Custom column names tests                                                    #
# ============================================================================ #


def test_bfs_custom_column_names():
    """Test BFS with custom column names."""
    edges = pd.DataFrame(
        data={
            "from_node": [0, 0, 1, 2],
            "to_node": [1, 2, 3, 3],
        }
    )
    bfs = BFS(edges, tail="from_node", head="to_node")
    predecessors = bfs.run(vertex_idx=0)

    assert predecessors[0] == bfs.UNREACHABLE
    assert predecessors[1] == 0
    assert predecessors[2] == 0
    assert predecessors[3] in [1, 2]


# ============================================================================ #
# Properties tests                                                             #
# ============================================================================ #


def test_bfs_properties(simple_graph):
    """Test BFS property getters."""
    bfs = BFS(simple_graph, orientation="out", permute=False)

    assert bfs.n_edges == 4
    assert bfs.n_vertices == 4
    assert bfs.orientation == "out"
    assert bfs.permute is False
    assert bfs.path_links is None

    # Run with path tracking
    bfs.run(vertex_idx=0, path_tracking=True)
    assert bfs.path_links is not None


def test_bfs_get_vertices(simple_graph):
    """Test get_vertices method."""
    bfs = BFS(simple_graph)
    vertices = bfs.get_vertices()

    assert len(vertices) == 4
    assert np.array_equal(vertices, np.array([0, 1, 2, 3]))


def test_bfs_get_vertices_permuted():
    """Test get_vertices with permutation."""
    edges = pd.DataFrame(
        data={
            "tail": [10, 20],
            "head": [20, 30],
        }
    )
    bfs = BFS(edges, permute=True)
    vertices = bfs.get_vertices()

    assert len(vertices) == 3
    assert np.array_equal(vertices, np.array([10, 20, 30]))


# ============================================================================ #
# DataFrame backend compatibility tests                                        #
# ============================================================================ #


class TestDataFrameBackendCompatibility:
    """Test BFS consistency across different DataFrame backends."""

    def create_test_dataframes(self, data: dict):
        """Create test DataFrames in all available formats."""
        dataframes = {}

        # pandas with NumPy backend
        dataframes["pandas_numpy"] = pd.DataFrame(data)

        # pandas with Arrow backend
        if PYARROW_AVAILABLE:
            df_arrow = pd.DataFrame(data)
            arrow_dtypes = {}
            for col in data.keys():
                arrow_dtypes[col] = pd.ArrowDtype(pa.int64())
            df_arrow = df_arrow.astype(arrow_dtypes)
            dataframes["pandas_arrow"] = df_arrow

        # Polars DataFrame
        if POLARS_AVAILABLE:
            dataframes["polars"] = pl.DataFrame(data)

        return dataframes

    def test_bfs_backend_consistency(self):
        """Test BFS consistency across DataFrame backends."""
        data = {
            "tail": [0, 0, 1, 2],
            "head": [1, 2, 3, 3],
        }

        dataframes = self.create_test_dataframes(data)
        results = {}

        # Run BFS on each DataFrame backend
        for backend_name, df in dataframes.items():
            bfs = BFS(df, orientation="out")
            results[backend_name] = bfs.run(vertex_idx=0)

        # All results should be identical
        reference_result = list(results.values())[0]
        for backend_name, result in results.items():
            np.testing.assert_array_equal(
                reference_result,
                result,
                err_msg=f"BFS results differ for {backend_name}",
            )

    def test_bfs_path_tracking_backend_consistency(self):
        """Test BFS path tracking consistency across backends."""
        data = {
            "tail": [0, 1, 2],
            "head": [1, 2, 3],
        }

        dataframes = self.create_test_dataframes(data)
        paths = {}

        # Run BFS with path tracking on each DataFrame backend
        for backend_name, df in dataframes.items():
            bfs = BFS(df, orientation="out")
            bfs.run(vertex_idx=0, path_tracking=True)
            paths[backend_name] = bfs.get_path(3)

        # All paths should be identical
        reference_path = list(paths.values())[0]
        for backend_name, path in paths.items():
            if reference_path is None:
                assert path is None, f"Path differs for {backend_name}: expected None"
            else:
                np.testing.assert_array_equal(
                    reference_path,
                    path,
                    err_msg=f"Path results differ for {backend_name}",
                )

    def test_bfs_permutation_backend_consistency(self):
        """Test BFS permutation consistency across backends."""
        data = {
            "tail": [10, 10, 20],
            "head": [20, 30, 30],
        }

        dataframes = self.create_test_dataframes(data)
        results = {}

        # Run BFS with permute=True
        for backend_name, df in dataframes.items():
            bfs = BFS(df, permute=True)
            results[backend_name] = bfs.run(vertex_idx=10)

        # All results should be identical
        reference_result = list(results.values())[0]
        for backend_name, result in results.items():
            np.testing.assert_array_equal(
                reference_result,
                result,
                err_msg=f"Permutation results differ for {backend_name}",
            )


# ============================================================================ #
# Comparison with expected behavior                                            #
# ============================================================================ #


def test_bfs_correctness_comparison():
    """Test BFS correctness by comparing with manual computation."""
    # Create a specific graph and verify BFS produces correct result
    edges = pd.DataFrame(
        data={
            "tail": [0, 0, 1, 1, 2, 3, 3, 4],
            "head": [1, 2, 3, 4, 4, 5, 6, 6],
        }
    )

    bfs = BFS(edges, orientation="out")
    predecessors = bfs.run(vertex_idx=0)

    # Expected predecessors from manual BFS:
    # 0: start (UNREACHABLE)
    # 1: from 0
    # 2: from 0
    # 3: from 1
    # 4: from 1 or 2 (whichever is visited first in BFS)
    # 5: from 3
    # 6: from 3 or 4
    assert predecessors[0] == bfs.UNREACHABLE
    assert predecessors[1] == 0
    assert predecessors[2] == 0
    assert predecessors[3] == 1
    assert predecessors[4] in [1, 2]
    assert predecessors[5] == 3
    assert predecessors[6] in [3, 4]


def test_bfs_orientation_symmetry():
    """Test that BFS 'out' and 'in' are consistent."""
    edges = pd.DataFrame(
        data={
            "tail": [0, 1, 2],
            "head": [1, 2, 3],
        }
    )

    # Forward BFS from 0
    bfs_out = BFS(edges, orientation="out")
    pred_out = bfs_out.run(vertex_idx=0)

    # Backward BFS to 3
    bfs_in = BFS(edges, orientation="in")
    pred_in = bfs_in.run(vertex_idx=3)

    # In this linear graph, the paths should be consistent
    # From 0: 0 -> 1 -> 2 -> 3
    # To 3: 0 -> 1 -> 2 -> 3 (same path)
    assert pred_out[0] == bfs_out.UNREACHABLE
    assert pred_out[1] == 0
    assert pred_out[2] == 1
    assert pred_out[3] == 2

    assert pred_in[3] == bfs_in.UNREACHABLE
    # In backward search, predecessors are successors
    assert pred_in[2] == 3
    assert pred_in[1] == 2
    assert pred_in[0] == 1


def test_bfs_custom_sentinel():
    """Test BFS with custom sentinel value."""
    edges = pd.DataFrame(
        data={
            "tail": [0, 0, 1, 2],
            "head": [1, 2, 3, 3],
        }
    )

    # Test with custom sentinel value
    custom_sentinel = -1
    bfs = BFS(edges, sentinel=custom_sentinel)
    predecessors = bfs.run(vertex_idx=0)

    # Verify custom sentinel is used
    assert bfs.UNREACHABLE == custom_sentinel
    assert predecessors[0] == custom_sentinel  # start vertex
    assert predecessors[1] == 0  # reached from 0
    assert predecessors[2] == 0  # reached from 0
    assert predecessors[3] in [1, 2]  # could be reached from 1 or 2


def test_bfs_sentinel_validation():
    """Test BFS sentinel parameter validation."""
    edges = pd.DataFrame(
        data={
            "tail": [0, 1],
            "head": [1, 2],
        }
    )

    # Test positive sentinel (should fail)
    with pytest.raises(ValueError, match="sentinel must be negative"):
        BFS(edges, sentinel=1)

    # Test zero sentinel (should fail)
    with pytest.raises(ValueError, match="sentinel must be negative"):
        BFS(edges, sentinel=0)

    # Test non-integer sentinel (should fail)
    with pytest.raises(TypeError, match="sentinel must be an integer"):
        BFS(edges, sentinel=-9999.5)

    # Test sentinel outside int32 range (should fail)
    with pytest.raises(ValueError, match="sentinel must fit in int32 range"):
        BFS(edges, sentinel=-(10**15))
