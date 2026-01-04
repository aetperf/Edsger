"""Test the GraphImporter factory method and backend selection logic."""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

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


class TestFactoryMethodDetection:
    """Test the GraphImporter factory method's DataFrame type detection."""

    def test_pandas_numpy_detection(self):
        """Test detection of pandas DataFrames with NumPy backend."""
        df = pd.DataFrame(
            {"tail": [0, 1, 2], "head": [1, 2, 3], "weight": [1.0, 2.0, 3.0]}
        )

        importer = GraphImporter.from_dataframe(df)
        assert isinstance(importer, PandasNumpyImporter)
        assert importer.edges_df is df

    @pytest.mark.skipif(not PYARROW_AVAILABLE, reason="PyArrow not available")
    def test_pandas_arrow_detection(self):
        """Test detection of pandas DataFrames with Arrow backend."""
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
        assert importer.edges_df is df

    @pytest.mark.skipif(not PYARROW_AVAILABLE, reason="PyArrow not available")
    def test_mixed_arrow_numpy_detection(self):
        """Test detection when only some columns are Arrow-backed."""
        df = pd.DataFrame(
            {"tail": [0, 1, 2], "head": [1, 2, 3], "weight": [1.0, 2.0, 3.0]}
        )

        # Convert only one column to Arrow
        df["tail"] = df["tail"].astype(pd.ArrowDtype(pa.int64()))

        importer = GraphImporter.from_dataframe(df)
        assert isinstance(importer, PandasArrowImporter)

    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not available")
    def test_polars_detection(self):
        """Test detection of Polars DataFrames."""
        df = pl.DataFrame(
            {"tail": [0, 1, 2], "head": [1, 2, 3], "weight": [1.0, 2.0, 3.0]}
        )

        importer = GraphImporter.from_dataframe(df)
        assert isinstance(importer, PolarsImporter)
        assert importer.edges_df is df

    def test_unknown_type_handling(self):
        """Test handling of unknown DataFrame types."""
        # Use a dict as an unknown type
        fake_df = {"tail": [0, 1, 2], "head": [1, 2, 3], "weight": [1.0, 2.0, 3.0]}

        with pytest.warns(UserWarning, match="Unknown DataFrame type"):
            importer = GraphImporter.from_dataframe(fake_df)

        # Should fall back to PandasNumpyImporter
        assert isinstance(importer, PandasNumpyImporter)

    def test_none_handling(self):
        """Test handling of None input."""
        # None gets converted to DataFrame, so we expect a warning and a PandasNumpyImporter
        with pytest.warns(UserWarning, match="Unknown DataFrame type"):
            importer = GraphImporter.from_dataframe(None)
        assert isinstance(importer, PandasNumpyImporter)

    def test_factory_parameters_passed(self):
        """Test that factory method parameters are passed correctly."""
        df = pd.DataFrame({"tail": [0, 1], "head": [1, 2], "weight": [1.0, 2.0]})

        # Parameters should be accepted (even if not used by all importers)
        importer = GraphImporter.from_dataframe(
            df,
            tail="tail",
            head="head",
            weight="weight",
            trav_time="trav_time",
            freq="freq",
        )
        assert isinstance(importer, PandasNumpyImporter)


class TestBackendSelectionLogic:
    """Test the logic behind backend selection."""

    @pytest.mark.skipif(not PYARROW_AVAILABLE, reason="PyArrow not available")
    def test_arrow_priority_over_numpy(self):
        """Test that Arrow backend is detected even with minimal Arrow columns."""
        df = pd.DataFrame(
            {
                "tail": [0, 1, 2],
                "head": [1, 2, 3],
                "weight": [1.0, 2.0, 3.0],
                "extra1": [10, 20, 30],
                "extra2": [100, 200, 300],
            }
        )

        # Convert only one column to Arrow
        df["extra1"] = df["extra1"].astype(pd.ArrowDtype(pa.int64()))

        importer = GraphImporter.from_dataframe(df)
        assert isinstance(importer, PandasArrowImporter)

    def test_polars_detection_priority(self):
        """Test that Polars detection takes priority."""
        if not POLARS_AVAILABLE:
            pytest.skip("Polars not available")

        # Create a Polars DataFrame
        df = pl.DataFrame(
            {"tail": [0, 1, 2], "head": [1, 2, 3], "weight": [1.0, 2.0, 3.0]}
        )

        importer = GraphImporter.from_dataframe(df)
        assert isinstance(importer, PolarsImporter)

    def test_module_name_detection(self):
        """Test that Polars is detected by module name."""
        if not POLARS_AVAILABLE:
            pytest.skip("Polars not available")

        df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})

        # Check that module detection works
        assert hasattr(df, "__class__")
        assert df.__class__.__module__.startswith("polars")

        importer = GraphImporter.from_dataframe(df)
        assert isinstance(importer, PolarsImporter)

    @pytest.mark.skipif(not PYARROW_AVAILABLE, reason="PyArrow not available")
    def test_pyarrow_dtype_detection(self):
        """Test the pyarrow_dtype attribute detection logic."""
        df = pd.DataFrame(
            {"tail": [0, 1, 2], "head": [1, 2, 3], "weight": [1.0, 2.0, 3.0]}
        )

        # Initially should be NumPy backend
        assert not any(hasattr(dtype, "pyarrow_dtype") for dtype in df.dtypes)
        importer = GraphImporter.from_dataframe(df)
        assert isinstance(importer, PandasNumpyImporter)

        # Convert to Arrow and check detection
        df["tail"] = df["tail"].astype(pd.ArrowDtype(pa.int64()))
        assert any(hasattr(dtype, "pyarrow_dtype") for dtype in df.dtypes)
        importer = GraphImporter.from_dataframe(df)
        assert isinstance(importer, PandasArrowImporter)


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases in backend selection."""

    def test_empty_dataframe(self):
        """Test handling of empty DataFrames."""
        df = pd.DataFrame(columns=["tail", "head", "weight"])

        importer = GraphImporter.from_dataframe(df)
        assert isinstance(importer, PandasNumpyImporter)

        result = importer.to_numpy_edges(["tail", "head", "weight"])
        assert len(result) == 0
        assert list(result.columns) == ["tail", "head", "weight"]

    @pytest.mark.filterwarnings("ignore:Unknown DataFrame type:UserWarning")
    def test_malformed_dataframe_object(self):
        """Test handling of objects that look like DataFrames but aren't."""

        class FakeDataFrame:
            def __init__(self):
                self.columns = ["tail", "head", "weight"]

            @property
            def __class__(self):
                # Return a class with a fake module
                class FakeClass:
                    __module__ = "fake_module"

                return FakeClass

        fake_df = FakeDataFrame()

        # This will fail when trying to convert to pandas DataFrame
        with pytest.raises(
            ValueError, match="DataFrame constructor not properly called"
        ):
            GraphImporter.from_dataframe(fake_df)

    @pytest.mark.skipif(not PYARROW_AVAILABLE, reason="PyArrow not available")
    def test_corrupted_arrow_dtype(self):
        """Test handling of corrupted Arrow dtype information."""
        df = pd.DataFrame(
            {"tail": [0, 1, 2], "head": [1, 2, 3], "weight": [1.0, 2.0, 3.0]}
        )

        # Manually corrupt the dtype to have pyarrow_dtype attribute but wrong type
        class FakeDtype:
            def __init__(self):
                self.pyarrow_dtype = "not_a_real_arrow_type"

        # This is artificial corruption - in real scenarios, this shouldn't happen
        # but we test defensive programming
        try:
            df["tail"] = df["tail"].astype(pd.ArrowDtype(pa.int64()))

            # Should still be detected as Arrow backend
            importer = GraphImporter.from_dataframe(df)
            assert isinstance(importer, PandasArrowImporter)
        finally:
            # Restore original state
            pass

    @pytest.mark.filterwarnings("ignore:Unknown DataFrame type:UserWarning")
    def test_class_attribute_access_error(self):
        """Test handling when __class__ or __module__ access fails."""

        class ProblematicDataFrame:
            @property
            def __class__(self):
                raise AttributeError("Can't access __class__")

        problematic_df = ProblematicDataFrame()

        # This will fail when trying to convert to pandas DataFrame
        with pytest.raises(
            ValueError, match="DataFrame constructor not properly called"
        ):
            GraphImporter.from_dataframe(problematic_df)


class TestImporterInstantiation:
    """Test that importers are instantiated correctly."""

    def test_importer_receives_dataframe(self):
        """Test that importers receive the correct DataFrame."""
        df = pd.DataFrame(
            {"tail": [0, 1, 2], "head": [1, 2, 3], "weight": [1.0, 2.0, 3.0]}
        )

        importer = GraphImporter.from_dataframe(df)
        assert importer.edges_df is df

    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not available")
    def test_polars_importer_instantiation(self):
        """Test Polars importer instantiation."""
        df = pl.DataFrame(
            {"tail": [0, 1, 2], "head": [1, 2, 3], "weight": [1.0, 2.0, 3.0]}
        )

        importer = GraphImporter.from_dataframe(df)
        assert isinstance(importer, PolarsImporter)
        assert importer.edges_df is df

    def test_multiple_instantiations(self):
        """Test that multiple instantiations work correctly."""
        df1 = pd.DataFrame({"tail": [0], "head": [1], "weight": [1.0]})
        df2 = pd.DataFrame({"tail": [1], "head": [2], "weight": [2.0]})

        importer1 = GraphImporter.from_dataframe(df1)
        importer2 = GraphImporter.from_dataframe(df2)

        assert importer1.edges_df is df1
        assert importer2.edges_df is df2
        assert importer1 is not importer2


class TestStandardizeFunctionIntegration:
    """Test integration with the standardize_graph_dataframe function."""

    def test_standardize_uses_factory(self):
        """Test that standardize function uses the factory method."""
        df = pd.DataFrame(
            {"tail": [0, 1, 2], "head": [1, 2, 3], "weight": [1.0, 2.0, 3.0]}
        )

        # Mock the factory to ensure it's called
        with patch.object(GraphImporter, "from_dataframe") as mock_factory:
            mock_importer = MagicMock()
            mock_importer.to_numpy_edges.return_value = pd.DataFrame(
                {"tail": [0, 1, 2], "head": [1, 2, 3], "weight": [1.0, 2.0, 3.0]}
            )
            mock_factory.return_value = mock_importer

            _ = standardize_graph_dataframe(df, "tail", "head", "weight")

            # Verify factory was called with correct parameters
            mock_factory.assert_called_once()
            args, kwargs = mock_factory.call_args
            assert args[0] is df

            # Verify to_numpy_edges was called
            mock_importer.to_numpy_edges.assert_called_once_with(
                ["tail", "head", "weight"]
            )

    def test_standardize_column_selection(self):
        """Test that standardize function correctly selects columns."""
        df = pd.DataFrame(
            {
                "tail": [0, 1, 2],
                "head": [1, 2, 3],
                "weight": [1.0, 2.0, 3.0],
                "extra": [10, 20, 30],
            }
        )

        result = standardize_graph_dataframe(df, "tail", "head", "weight")

        # Should only have the requested columns
        assert list(result.columns) == ["tail", "head", "weight"]
        assert "extra" not in result.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
