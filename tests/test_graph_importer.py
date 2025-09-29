"""Test GraphImporter functionality."""

import pytest
import numpy as np
import pandas as pd

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

from edsger.graph_importer import (
    GraphImporter,
    PandasNumpyImporter,
    PandasArrowImporter,
    PolarsImporter,
    standardize_graph_dataframe,
)


class TestGraphImporterFactory:
    """Test the factory method for creating appropriate importers."""

    def test_pandas_numpy_detection(self):
        """Test that NumPy-backed pandas DataFrames are correctly detected."""
        df = pd.DataFrame(
            {"tail": [0, 1, 2], "head": [1, 2, 3], "weight": [1.0, 2.0, 3.0]}
        )

        importer = GraphImporter.from_dataframe(df)
        assert isinstance(importer, PandasNumpyImporter)

    @pytest.mark.skipif(not PYARROW_AVAILABLE, reason="PyArrow not available")
    def test_pandas_arrow_detection(self):
        """Test that Arrow-backed pandas DataFrames are correctly detected."""
        df = pd.DataFrame(
            {"tail": [0, 1, 2], "head": [1, 2, 3], "weight": [1.0, 2.0, 3.0]}
        )
        # Convert to Arrow backend
        df = df.astype(
            {
                "tail": pd.ArrowDtype(pa.int64()),
                "head": pd.ArrowDtype(pa.int64()),
                "weight": pd.ArrowDtype(pa.float64()),
            }
        )

        importer = GraphImporter.from_dataframe(df)
        assert isinstance(importer, PandasArrowImporter)

    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not available")
    def test_polars_detection(self):
        """Test that Polars DataFrames are correctly detected."""
        df = pl.DataFrame(
            {"tail": [0, 1, 2], "head": [1, 2, 3], "weight": [1.0, 2.0, 3.0]}
        )

        importer = GraphImporter.from_dataframe(df)
        assert isinstance(importer, PolarsImporter)


class TestPandasNumpyImporter:
    """Test PandasNumpyImporter functionality."""

    def test_to_numpy_edges(self):
        """Test conversion for NumPy-backed pandas."""
        df = pd.DataFrame(
            {
                "tail": [0, 1, 2],
                "head": [1, 2, 3],
                "weight": [1.0, 2.0, 3.0],
                "extra": [10, 20, 30],  # Extra column that should be ignored
            }
        )

        importer = PandasNumpyImporter(df)
        result = importer.to_numpy_edges(["tail", "head", "weight"])

        # Check that result has correct columns
        assert list(result.columns) == ["tail", "head", "weight"]
        assert len(result) == 3

        # Check that values are preserved
        np.testing.assert_array_equal(result["tail"].values, [0, 1, 2])
        np.testing.assert_array_equal(result["head"].values, [1, 2, 3])
        np.testing.assert_array_equal(result["weight"].values, [1.0, 2.0, 3.0])

        # Check that arrays are contiguous
        assert result["tail"].values.flags["C_CONTIGUOUS"]
        assert result["head"].values.flags["C_CONTIGUOUS"]
        assert result["weight"].values.flags["C_CONTIGUOUS"]


class TestPandasArrowImporter:
    """Test PandasArrowImporter functionality."""

    @pytest.mark.skipif(not PYARROW_AVAILABLE, reason="PyArrow not available")
    def test_to_numpy_edges(self):
        """Test conversion for Arrow-backed pandas."""
        df = pd.DataFrame(
            {"tail": [0, 1, 2], "head": [1, 2, 3], "weight": [1.0, 2.0, 3.0]}
        )
        # Convert to Arrow backend
        df = df.astype(
            {
                "tail": pd.ArrowDtype(pa.int64()),
                "head": pd.ArrowDtype(pa.int64()),
                "weight": pd.ArrowDtype(pa.float64()),
            }
        )

        importer = PandasArrowImporter(df)
        result = importer.to_numpy_edges(["tail", "head", "weight"])

        # Check that result has correct columns
        assert list(result.columns) == ["tail", "head", "weight"]
        assert len(result) == 3

        # Check that values are preserved and converted to appropriate types
        assert result["tail"].dtype == np.uint32  # Vertex indices converted to uint32
        assert result["head"].dtype == np.uint32  # Vertex indices converted to uint32
        assert result["weight"].dtype == np.float64

        np.testing.assert_array_equal(result["tail"].values, [0, 1, 2])
        np.testing.assert_array_equal(result["head"].values, [1, 2, 3])
        np.testing.assert_array_equal(result["weight"].values, [1.0, 2.0, 3.0])

        # Check that arrays are contiguous
        assert result["tail"].values.flags["C_CONTIGUOUS"]
        assert result["head"].values.flags["C_CONTIGUOUS"]
        assert result["weight"].values.flags["C_CONTIGUOUS"]

    @pytest.mark.skipif(not PYARROW_AVAILABLE, reason="PyArrow not available")
    def test_large_vertex_indices(self):
        """Test that large vertex indices use uint64."""
        large_val = np.iinfo(np.uint32).max + 1
        df = pd.DataFrame(
            {"tail": [0, 1, large_val], "head": [1, 2, 3], "weight": [1.0, 2.0, 3.0]}
        )
        # Convert to Arrow backend
        df = df.astype(
            {
                "tail": pd.ArrowDtype(pa.int64()),
                "head": pd.ArrowDtype(pa.int64()),
                "weight": pd.ArrowDtype(pa.float64()),
            }
        )

        importer = PandasArrowImporter(df)
        result = importer.to_numpy_edges(["tail", "head", "weight"])

        # Should use uint64 for large indices
        assert result["tail"].dtype == np.uint64


class TestPolarsImporter:
    """Test PolarsImporter functionality."""

    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not available")
    def test_to_numpy_edges(self):
        """Test conversion for Polars DataFrames."""
        df = pl.DataFrame(
            {"tail": [0, 1, 2], "head": [1, 2, 3], "weight": [1.0, 2.0, 3.0]}
        )

        importer = PolarsImporter(df)
        result = importer.to_numpy_edges(["tail", "head", "weight"])

        # Check that result is a pandas DataFrame
        assert isinstance(result, pd.DataFrame)

        # Check that result has correct columns
        assert list(result.columns) == ["tail", "head", "weight"]
        assert len(result) == 3

        # Check that values are preserved and converted to appropriate types
        assert result["tail"].dtype == np.uint32  # Vertex indices converted to uint32
        assert result["head"].dtype == np.uint32  # Vertex indices converted to uint32
        assert result["weight"].dtype == np.float64

        np.testing.assert_array_equal(result["tail"].values, [0, 1, 2])
        np.testing.assert_array_equal(result["head"].values, [1, 2, 3])
        np.testing.assert_array_equal(result["weight"].values, [1.0, 2.0, 3.0])

        # Check that arrays are contiguous
        assert result["tail"].values.flags["C_CONTIGUOUS"]
        assert result["head"].values.flags["C_CONTIGUOUS"]
        assert result["weight"].values.flags["C_CONTIGUOUS"]


class TestStandardizeFunction:
    """Test the standardize_graph_dataframe convenience function."""

    def test_standardize_numpy_backend(self):
        """Test standardization with NumPy backend."""
        df = pd.DataFrame(
            {
                "tail": [0, 1, 2],
                "head": [1, 2, 3],
                "weight": [1.0, 2.0, 3.0],
                "extra": [10, 20, 30],
            }
        )

        result = standardize_graph_dataframe(df, "tail", "head", "weight")

        # Should only have requested columns
        assert list(result.columns) == ["tail", "head", "weight"]
        assert len(result) == 3

        # Values should be preserved
        np.testing.assert_array_equal(result["tail"].values, [0, 1, 2])
        np.testing.assert_array_equal(result["head"].values, [1, 2, 3])
        np.testing.assert_array_equal(result["weight"].values, [1.0, 2.0, 3.0])

    @pytest.mark.skipif(not PYARROW_AVAILABLE, reason="PyArrow not available")
    def test_standardize_arrow_backend(self):
        """Test standardization with Arrow backend."""
        df = pd.DataFrame(
            {"tail": [0, 1, 2], "head": [1, 2, 3], "weight": [1.0, 2.0, 3.0]}
        )
        df = df.astype(
            {
                "tail": pd.ArrowDtype(pa.int64()),
                "head": pd.ArrowDtype(pa.int64()),
                "weight": pd.ArrowDtype(pa.float64()),
            }
        )

        result = standardize_graph_dataframe(df, "tail", "head", "weight")

        # Should be converted to NumPy backend
        assert not hasattr(result["tail"].dtype, "pyarrow_dtype")
        assert result["tail"].dtype == np.uint32
        assert result["head"].dtype == np.uint32
        assert result["weight"].dtype == np.float64

    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not available")
    def test_standardize_polars(self):
        """Test standardization with Polars DataFrame."""
        df = pl.DataFrame(
            {"tail": [0, 1, 2], "head": [1, 2, 3], "weight": [1.0, 2.0, 3.0]}
        )

        result = standardize_graph_dataframe(df, "tail", "head", "weight")

        # Should be converted to pandas with NumPy backend
        assert isinstance(result, pd.DataFrame)
        assert result["tail"].dtype == np.uint32
        assert result["head"].dtype == np.uint32
        assert result["weight"].dtype == np.float64

    def test_standardize_with_hyperpath_columns(self):
        """Test standardization with trav_time and freq columns for hyperpath."""
        df = pd.DataFrame(
            {
                "tail": [0, 1, 2],
                "head": [1, 2, 3],
                "trav_time": [1.0, 2.0, 3.0],
                "freq": [0.1, 0.2, 0.3],
            }
        )

        result = standardize_graph_dataframe(
            df, "tail", "head", trav_time="trav_time", freq="freq"
        )

        # Should have all requested columns
        assert list(result.columns) == ["tail", "head", "trav_time", "freq"]
        assert len(result) == 3


class TestEdgeCases:
    """Test edge cases and error handling in GraphImporter."""

    def test_empty_dataframe(self):
        """Test handling of empty DataFrames."""
        df = pd.DataFrame(columns=["tail", "head", "weight"])

        result = standardize_graph_dataframe(df, "tail", "head", "weight")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert list(result.columns) == ["tail", "head", "weight"]

    def test_single_row_dataframe(self):
        """Test handling of single-row DataFrames."""
        df = pd.DataFrame({"tail": [0], "head": [1], "weight": [1.5]})

        result = standardize_graph_dataframe(df, "tail", "head", "weight")

        assert len(result) == 1
        assert result["tail"].iloc[0] == 0
        assert result["head"].iloc[0] == 1
        assert result["weight"].iloc[0] == 1.5

    def test_unsupported_dataframe_type(self):
        """Test warning for unsupported DataFrame types."""
        # Use a dict as unsupported type
        fake_df = {"tail": [0, 1], "head": [1, 2], "weight": [1.0, 2.0]}

        with pytest.warns(UserWarning, match="Unknown DataFrame type"):
            result = standardize_graph_dataframe(fake_df, "tail", "head", "weight")

        # Should still work by converting to pandas
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    @pytest.mark.skipif(not PYARROW_AVAILABLE, reason="PyArrow not available")
    def test_mixed_arrow_numpy_columns(self):
        """Test DataFrame with mixed Arrow and NumPy columns."""
        df = pd.DataFrame(
            {"tail": [0, 1, 2], "head": [1, 2, 3], "weight": [1.0, 2.0, 3.0]}
        )

        # Convert only some columns to Arrow
        df["tail"] = df["tail"].astype(pd.ArrowDtype(pa.int64()))
        # head stays NumPy, weight stays NumPy

        # Should still be detected as Arrow backend
        importer = GraphImporter.from_dataframe(df)
        assert isinstance(importer, PandasArrowImporter)

        result = importer.to_numpy_edges(["tail", "head", "weight"])

        # All should be converted to NumPy
        assert all(isinstance(result[col].values, np.ndarray) for col in result.columns)

    def test_memory_contiguity(self):
        """Test that all returned arrays are C-contiguous."""
        df = pd.DataFrame(
            {
                "tail": [0, 1, 2, 3, 4],
                "head": [1, 2, 3, 4, 5],
                "weight": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )

        result = standardize_graph_dataframe(df, "tail", "head", "weight")

        # Check that all arrays are C-contiguous
        for col in result.columns:
            assert result[col].values.flags[
                "C_CONTIGUOUS"
            ], f"Column {col} is not C-contiguous"


class TestPerformanceAndMemory:
    """Test performance and memory characteristics."""

    def test_dtype_optimization(self):
        """Test that appropriate dtypes are chosen for memory efficiency."""
        # For NumPy-backed pandas, dtypes are preserved (no automatic optimization)
        small_df = pd.DataFrame(
            {"tail": [0, 1, 2], "head": [1, 2, 3], "weight": [1.0, 2.0, 3.0]}
        )

        result = standardize_graph_dataframe(small_df, "tail", "head", "weight")

        # NumPy backend preserves original dtypes
        assert result["tail"].dtype == np.int64  # pandas default
        assert result["head"].dtype == np.int64  # pandas default
        assert result["weight"].dtype == np.float64

    @pytest.mark.skipif(not PYARROW_AVAILABLE, reason="PyArrow not available")
    def test_large_vertex_indices_with_arrow(self):
        """Test that large vertex indices correctly use uint64."""
        large_val = np.iinfo(np.uint32).max + 10

        df = pd.DataFrame(
            {"tail": [0, 1, large_val], "head": [1, 2, 3], "weight": [1.0, 2.0, 3.0]}
        )
        df = df.astype(
            {
                "tail": pd.ArrowDtype(pa.int64()),
                "head": pd.ArrowDtype(pa.int64()),
                "weight": pd.ArrowDtype(pa.float64()),
            }
        )

        importer = PandasArrowImporter(df)
        result = importer.to_numpy_edges(["tail", "head", "weight"])

        # Should use uint64 for large indices
        assert result["tail"].dtype == np.uint64
        assert result["head"].dtype == np.uint32  # head column has small values

    def test_conversion_consistency(self):
        """Test that multiple conversions of the same data yield identical results."""
        df = pd.DataFrame(
            {"tail": [0, 1, 2, 3], "head": [1, 2, 3, 0], "weight": [1.5, 2.5, 3.5, 4.5]}
        )

        result1 = standardize_graph_dataframe(df, "tail", "head", "weight")
        result2 = standardize_graph_dataframe(df.copy(), "tail", "head", "weight")

        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)

    @pytest.mark.skipif(not PYARROW_AVAILABLE, reason="PyArrow not available")
    def test_arrow_to_numpy_performance(self):
        """Test that Arrow to NumPy conversion preserves data correctly."""
        # Create larger dataset for more realistic test
        n = 1000
        df = pd.DataFrame(
            {
                "tail": np.random.randint(0, 100, n),
                "head": np.random.randint(0, 100, n),
                "weight": np.random.random(n),
            }
        )

        # Convert to Arrow
        df_arrow = df.astype(
            {
                "tail": pd.ArrowDtype(pa.int64()),
                "head": pd.ArrowDtype(pa.int64()),
                "weight": pd.ArrowDtype(pa.float64()),
            }
        )

        # Convert back via GraphImporter
        result = standardize_graph_dataframe(df_arrow, "tail", "head", "weight")

        # Should have same values (allowing for dtype conversion)
        np.testing.assert_array_equal(
            result["tail"].values, df["tail"].values.astype(np.uint32)
        )
        np.testing.assert_array_equal(
            result["head"].values, df["head"].values.astype(np.uint32)
        )
        np.testing.assert_array_equal(result["weight"].values, df["weight"].values)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
