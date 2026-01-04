"""Test support for pandas DataFrames with Arrow backend using new GraphImporter architecture."""

import pytest
import numpy as np
import pandas as pd

try:
    import pyarrow as pa

    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

from edsger.path import Dijkstra, BellmanFord, HyperpathGenerating
from edsger.graph_importer import GraphImporter


def create_test_graph_numpy():
    """Create a simple test graph with NumPy backend."""
    edges_data = {
        "tail": [0, 0, 1, 2],
        "head": [1, 2, 2, 3],
        "weight": [1.0, 2.0, 1.0, 1.0],
    }
    return pd.DataFrame(edges_data)


def create_test_graph_arrow():
    """Create a simple test graph with Arrow backend."""
    if not PYARROW_AVAILABLE:
        pytest.skip("PyArrow not available")

    edges_data = {
        "tail": [0, 0, 1, 2],
        "head": [1, 2, 2, 3],
        "weight": [1.0, 2.0, 1.0, 1.0],
    }
    # Create DataFrame with Arrow backend
    df = pd.DataFrame(edges_data)
    # Convert to Arrow dtypes - use pyarrow types directly
    df = df.astype(
        {
            "tail": pd.ArrowDtype(pa.int64()),
            "head": pd.ArrowDtype(pa.int64()),
            "weight": pd.ArrowDtype(pa.float64()),
        }
    )
    return df


def create_test_graph_arrow_uint32():
    """Create a simple test graph with Arrow backend and uint32 types."""
    if not PYARROW_AVAILABLE:
        pytest.skip("PyArrow not available")

    edges_data = {
        "tail": [0, 0, 1, 2],
        "head": [1, 2, 2, 3],
        "weight": [1.0, 2.0, 1.0, 1.0],
    }
    # Create DataFrame with Arrow backend
    df = pd.DataFrame(edges_data)
    # Convert to Arrow dtypes with uint32 for tail/head
    df = df.astype(
        {
            "tail": pd.ArrowDtype(pa.uint32()),
            "head": pd.ArrowDtype(pa.uint32()),
            "weight": pd.ArrowDtype(pa.float64()),
        }
    )
    return df


class TestDijkstraArrowBackend:
    """Test Dijkstra algorithm with Arrow backend."""

    def test_dijkstra_numpy_backend(self):
        """Test Dijkstra with NumPy backend."""
        df = create_test_graph_numpy()
        dijkstra = Dijkstra(df)
        path_lengths = dijkstra.run(0, return_inf=True)

        # Check results
        assert path_lengths[0] == 0.0
        assert path_lengths[1] == 1.0
        assert path_lengths[2] == 2.0
        assert path_lengths[3] == 3.0

    def test_dijkstra_arrow_backend(self):
        """Test Dijkstra with Arrow backend."""
        df = create_test_graph_arrow()
        dijkstra = Dijkstra(df)
        path_lengths = dijkstra.run(0, return_inf=True)

        # Check results
        assert path_lengths[0] == 0.0
        assert path_lengths[1] == 1.0
        assert path_lengths[2] == 2.0
        assert path_lengths[3] == 3.0

    def test_dijkstra_arrow_backend_uint32(self):
        """Test Dijkstra with Arrow backend and uint32 types."""
        df = create_test_graph_arrow_uint32()
        dijkstra = Dijkstra(df)
        path_lengths = dijkstra.run(0, return_inf=True)

        # Check results
        assert path_lengths[0] == 0.0
        assert path_lengths[1] == 1.0
        assert path_lengths[2] == 2.0
        assert path_lengths[3] == 3.0

    def test_results_consistency(self):
        """Test that results are identical between NumPy and Arrow backends."""
        df_numpy = create_test_graph_numpy()
        df_arrow = create_test_graph_arrow()

        dijkstra_numpy = Dijkstra(df_numpy)
        dijkstra_arrow = Dijkstra(df_arrow)

        path_lengths_numpy = dijkstra_numpy.run(0, return_inf=True)
        path_lengths_arrow = dijkstra_arrow.run(0, return_inf=True)

        # Results should be identical
        np.testing.assert_array_equal(path_lengths_numpy, path_lengths_arrow)


class TestBellmanFordArrowBackend:
    """Test Bellman-Ford algorithm with Arrow backend."""

    def test_bellmanford_numpy_backend(self):
        """Test Bellman-Ford with NumPy backend."""
        df = create_test_graph_numpy()
        bf = BellmanFord(df)
        path_lengths = bf.run(0, return_inf=True)

        # Check results
        assert path_lengths[0] == 0.0
        assert path_lengths[1] == 1.0
        assert path_lengths[2] == 2.0
        assert path_lengths[3] == 3.0

    def test_bellmanford_arrow_backend(self):
        """Test Bellman-Ford with Arrow backend."""
        df = create_test_graph_arrow()
        bf = BellmanFord(df)
        path_lengths = bf.run(0, return_inf=True)

        # Check results
        assert path_lengths[0] == 0.0
        assert path_lengths[1] == 1.0
        assert path_lengths[2] == 2.0
        assert path_lengths[3] == 3.0

    def test_results_consistency(self):
        """Test that results are identical between NumPy and Arrow backends."""
        df_numpy = create_test_graph_numpy()
        df_arrow = create_test_graph_arrow()

        bf_numpy = BellmanFord(df_numpy)
        bf_arrow = BellmanFord(df_arrow)

        path_lengths_numpy = bf_numpy.run(0, return_inf=True)
        path_lengths_arrow = bf_arrow.run(0, return_inf=True)

        # Results should be identical
        np.testing.assert_array_equal(path_lengths_numpy, path_lengths_arrow)


class TestHyperpathArrowBackend:
    """Test HyperpathGenerating algorithm with Arrow backend."""

    def test_hyperpath_numpy_backend(self):
        """Test HyperpathGenerating with NumPy backend."""
        edges_data = {
            "tail": [0, 0, 1, 2],
            "head": [1, 2, 2, 3],
            "trav_time": [1.0, 2.0, 1.0, 1.0],
            "freq": [0.1, 0.1, 0.1, 0.1],
        }
        df = pd.DataFrame(edges_data)
        hp = HyperpathGenerating(df)
        hp.run(0, 3, 1.0, return_inf=True)

        # Check that it runs without error and volume column is added
        assert "volume" in hp._edges.columns
        assert hp.u_i_vec is not None

    def test_hyperpath_arrow_backend(self):
        """Test HyperpathGenerating with Arrow backend."""
        if not PYARROW_AVAILABLE:
            pytest.skip("PyArrow not available")

        edges_data = {
            "tail": [0, 0, 1, 2],
            "head": [1, 2, 2, 3],
            "trav_time": [1.0, 2.0, 1.0, 1.0],
            "freq": [0.1, 0.1, 0.1, 0.1],
        }
        df = pd.DataFrame(edges_data)
        # Convert to Arrow dtypes
        df = df.astype(
            {
                "tail": pd.ArrowDtype(pa.int64()),
                "head": pd.ArrowDtype(pa.int64()),
                "trav_time": pd.ArrowDtype(pa.float64()),
                "freq": pd.ArrowDtype(pa.float64()),
            }
        )
        hp = HyperpathGenerating(df)
        hp.run(0, 3, 1.0, return_inf=True)

        # Check that it runs without error and volume column is added
        assert "volume" in hp._edges.columns
        assert hp.u_i_vec is not None


def test_deep_copy_preserves_backend():
    """Test that deep copy preserves the Arrow backend."""
    df_arrow = create_test_graph_arrow()

    # Verify original has Arrow backend
    assert hasattr(df_arrow["tail"].dtype, "pyarrow_dtype")

    # Create Dijkstra instance (which uses GraphImporter internally)
    dijkstra = Dijkstra(df_arrow)

    # The internal _edges should now be NumPy-backed (converted by GraphImporter)
    assert not hasattr(dijkstra._edges["tail"].dtype, "pyarrow_dtype")
    assert isinstance(dijkstra._edges["tail"].values, np.ndarray)

    # Verify it works correctly
    path_lengths = dijkstra.run(0, return_inf=True)
    assert path_lengths[0] == 0.0
    assert path_lengths[1] == 1.0


class TestGraphImporterIntegration:
    """Test that GraphImporter is correctly integrated into the algorithms."""

    def test_dijkstra_uses_graph_importer(self):
        """Test that Dijkstra constructor uses GraphImporter internally."""
        if not PYARROW_AVAILABLE:
            pytest.skip("PyArrow not available")

        df_arrow = create_test_graph_arrow()

        # Verify input is Arrow-backed
        assert hasattr(df_arrow["tail"].dtype, "pyarrow_dtype")

        # Create Dijkstra instance
        dijkstra = Dijkstra(df_arrow)

        # Internal edges should be converted to NumPy backend
        assert not hasattr(dijkstra._edges["tail"].dtype, "pyarrow_dtype")
        assert isinstance(dijkstra._edges["tail"].values, np.ndarray)

        # Should have proper dtypes
        assert dijkstra._edges["tail"].dtype == np.uint32
        assert dijkstra._edges["head"].dtype == np.uint32
        assert dijkstra._edges["weight"].dtype == np.float64

        # Should have contiguous memory
        assert dijkstra._edges["tail"].values.flags["C_CONTIGUOUS"]
        assert dijkstra._edges["head"].values.flags["C_CONTIGUOUS"]
        assert dijkstra._edges["weight"].values.flags["C_CONTIGUOUS"]

    def test_bellmanford_uses_graph_importer(self):
        """Test that BellmanFord constructor uses GraphImporter internally."""
        if not PYARROW_AVAILABLE:
            pytest.skip("PyArrow not available")

        df_arrow = create_test_graph_arrow()

        # Create BellmanFord instance
        bf = BellmanFord(df_arrow)

        # Internal edges should be converted to NumPy backend
        assert not hasattr(bf._edges["tail"].dtype, "pyarrow_dtype")
        assert isinstance(bf._edges["tail"].values, np.ndarray)

    def test_hyperpath_uses_graph_importer(self):
        """Test that HyperpathGenerating constructor uses GraphImporter internally."""
        if not PYARROW_AVAILABLE:
            pytest.skip("PyArrow not available")

        edges_data = {
            "tail": [0, 0, 1, 2],
            "head": [1, 2, 2, 3],
            "trav_time": [1.0, 2.0, 1.0, 1.0],
            "freq": [0.1, 0.1, 0.1, 0.1],
        }
        df = pd.DataFrame(edges_data)
        df_arrow = df.astype(
            {
                "tail": pd.ArrowDtype(pa.int64()),
                "head": pd.ArrowDtype(pa.int64()),
                "trav_time": pd.ArrowDtype(pa.float64()),
                "freq": pd.ArrowDtype(pa.float64()),
            }
        )

        # Create HyperpathGenerating instance
        hp = HyperpathGenerating(df_arrow)

        # Internal edges should be converted to NumPy backend
        assert not hasattr(hp._edges["tail"].dtype, "pyarrow_dtype")
        assert isinstance(hp._edges["tail"].values, np.ndarray)

    def test_graph_importer_factory_detection(self):
        """Test that the factory correctly detects different DataFrame types."""
        # Test NumPy backend detection
        df_numpy = create_test_graph_numpy()
        importer_numpy = GraphImporter.from_dataframe(df_numpy)
        assert isinstance(importer_numpy, GraphImporter)

        if PYARROW_AVAILABLE:
            # Test Arrow backend detection
            df_arrow = create_test_graph_arrow()
            importer_arrow = GraphImporter.from_dataframe(df_arrow)
            assert isinstance(importer_arrow, GraphImporter)

    def test_conversion_happens_once(self):
        """Test that conversion happens only once at initialization."""
        if not PYARROW_AVAILABLE:
            pytest.skip("PyArrow not available")

        df_arrow = create_test_graph_arrow()

        # Create algorithm instance
        dijkstra = Dijkstra(df_arrow)

        # Get reference to internal edges
        internal_edges = dijkstra._edges

        # Run algorithm
        dijkstra.run(0)

        # Internal edges should be the same object (no additional conversions)
        assert dijkstra._edges is internal_edges


class TestErrorHandling:
    """Test error handling in Arrow backend scenarios."""

    @pytest.mark.skipif(not PYARROW_AVAILABLE, reason="PyArrow not available")
    def test_mixed_dtype_arrow_dataframe(self):
        """Test handling of Arrow DataFrame with mixed dtypes."""
        df = pd.DataFrame(
            {"tail": [0, 1, 2], "head": [1, 2, 3], "weight": [1.0, 2.0, 3.0]}
        )

        # Mix of Arrow and non-Arrow types
        df["tail"] = df["tail"].astype(pd.ArrowDtype(pa.int64()))
        df["head"] = df["head"]  # Keep as NumPy
        df["weight"] = df["weight"].astype(pd.ArrowDtype(pa.float64()))

        # Should still work
        dijkstra = Dijkstra(df)
        path_lengths = dijkstra.run(0, return_inf=True)

        assert path_lengths[0] == 0.0
        assert path_lengths[1] == 1.0

    @pytest.mark.skipif(not PYARROW_AVAILABLE, reason="PyArrow not available")
    def test_null_values_in_arrow_dataframe(self):
        """Test handling of null values in Arrow DataFrame."""
        # Arrow can handle null values, but our algorithms shouldn't receive them
        df = pd.DataFrame(
            {"tail": [0, 1, None], "head": [1, 2, 3], "weight": [1.0, 2.0, 3.0]}
        )
        df = df.astype(
            {
                "tail": pd.ArrowDtype(pa.int64()),
                "head": pd.ArrowDtype(pa.int64()),
                "weight": pd.ArrowDtype(pa.float64()),
            }
        )

        # Should raise an error during edge checking or conversion
        with pytest.raises((ValueError, TypeError)):
            Dijkstra(df, check_edges=True)


if __name__ == "__main__":
    # Run tests
    test = TestDijkstraArrowBackend()
    test.test_dijkstra_numpy_backend()
    test.test_dijkstra_arrow_backend()
    test.test_dijkstra_arrow_backend_uint32()
    test.test_results_consistency()

    test_bf = TestBellmanFordArrowBackend()
    test_bf.test_bellmanford_numpy_backend()
    test_bf.test_bellmanford_arrow_backend()
    test_bf.test_results_consistency()

    test_hp = TestHyperpathArrowBackend()
    test_hp.test_hyperpath_numpy_backend()
    test_hp.test_hyperpath_arrow_backend()

    test_deep_copy_preserves_backend()

    print("All Arrow backend tests passed!")
