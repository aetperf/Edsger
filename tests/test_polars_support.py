"""Test Polars DataFrame support in graph algorithms."""

import pytest
import numpy as np
import pandas as pd

try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

from edsger.path import Dijkstra, BellmanFord


@pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not available")
class TestPolarsSupport:
    """Test that graph algorithms work with Polars DataFrames."""

    def create_polars_graph(self):
        """Create a simple test graph using Polars."""
        return pl.DataFrame(
            {"tail": [0, 0, 1, 2], "head": [1, 2, 2, 3], "weight": [1.0, 4.0, 2.0, 1.0]}
        )

    def test_dijkstra_with_polars(self):
        """Test Dijkstra algorithm with Polars DataFrame."""
        df = self.create_polars_graph()

        # Should work directly with Polars DataFrame
        dijkstra = Dijkstra(df)
        path_lengths = dijkstra.run(0, return_inf=True)

        # Verify results
        assert path_lengths[0] == 0.0
        assert path_lengths[1] == 1.0
        assert path_lengths[2] == 3.0
        assert path_lengths[3] == 4.0

    def test_bellmanford_with_polars(self):
        """Test Bellman-Ford algorithm with Polars DataFrame."""
        df = self.create_polars_graph()

        # Should work directly with Polars DataFrame
        bf = BellmanFord(df)
        path_lengths = bf.run(0, return_inf=True)

        # Verify results
        assert path_lengths[0] == 0.0
        assert path_lengths[1] == 1.0
        assert path_lengths[2] == 3.0
        assert path_lengths[3] == 4.0

    def test_polars_with_custom_column_names(self):
        """Test with custom column names in Polars DataFrame."""
        df = pl.DataFrame(
            {
                "from_node": [0, 0, 1, 2],
                "to_node": [1, 2, 2, 3],
                "cost": [1.0, 4.0, 2.0, 1.0],
            }
        )

        # Should work with custom column names
        dijkstra = Dijkstra(df, tail="from_node", head="to_node", weight="cost")
        path_lengths = dijkstra.run(0, return_inf=True)

        assert path_lengths[0] == 0.0
        assert path_lengths[1] == 1.0
        assert path_lengths[2] == 3.0
        assert path_lengths[3] == 4.0

    def test_large_polars_graph(self):
        """Test with a larger Polars graph."""
        # Create a larger graph
        n = 100
        edges = []
        for i in range(n - 1):
            edges.append((i, i + 1, 1.0))  # Linear chain
            if i % 10 == 0 and i + 10 < n:
                edges.append((i, i + 10, 8.0))  # Shortcuts

        tail, head, weight = zip(*edges)
        df = pl.DataFrame({"tail": tail, "head": head, "weight": weight})

        dijkstra = Dijkstra(df)
        path_lengths = dijkstra.run(0, return_inf=True)

        # Check some path lengths
        assert path_lengths[0] == 0.0
        assert path_lengths[10] == 8.0  # Should use shortcut
        assert path_lengths[20] == 16.0  # Should use two shortcuts

    def test_polars_memory_efficiency(self):
        """Test that Polars DataFrames are converted efficiently."""
        # Create DataFrame with specific dtypes
        df = pl.DataFrame(
            {
                "tail": pl.Series([0, 0, 1, 2], dtype=pl.UInt32),
                "head": pl.Series([1, 2, 2, 3], dtype=pl.UInt32),
                "weight": pl.Series([1.0, 4.0, 2.0, 1.0], dtype=pl.Float64),
            }
        )

        dijkstra = Dijkstra(df)

        # Internal representation should use appropriate dtypes
        assert dijkstra._edges["tail"].dtype == np.uint32
        assert dijkstra._edges["head"].dtype == np.uint32
        assert dijkstra._edges["weight"].dtype == np.float64

        # Should have contiguous memory
        assert dijkstra._edges["tail"].values.flags["C_CONTIGUOUS"]

    def test_polars_to_pandas_consistency(self):
        """Test that Polars and pandas produce identical results."""
        # Create same data in both formats
        data = {
            "tail": [0, 0, 1, 2, 3],
            "head": [1, 2, 2, 3, 4],
            "weight": [1.0, 4.0, 2.0, 1.0, 3.0],
        }

        df_pandas = pd.DataFrame(data)
        df_polars = pl.DataFrame(data)

        # Run algorithms
        dijkstra_pandas = Dijkstra(df_pandas)
        dijkstra_polars = Dijkstra(df_polars)

        paths_pandas = dijkstra_pandas.run(0, return_inf=True)
        paths_polars = dijkstra_polars.run(0, return_inf=True)

        # Results should be identical
        np.testing.assert_array_equal(paths_pandas, paths_polars)

    def test_polars_string_columns(self):
        """Test that Polars DataFrames require numeric vertex indices."""
        df = pl.DataFrame(
            {
                "source": ["A", "A", "B", "C"],
                "target": ["B", "C", "C", "D"],
                "cost": [1.0, 4.0, 2.0, 1.0],
            }
        )

        # String columns for vertices are not supported - should raise TypeError
        with pytest.raises(TypeError):
            Dijkstra(df, tail="source", head="target", weight="cost")

        # Convert to numeric indices first
        df_numeric = pl.DataFrame(
            {
                "source": [0, 0, 1, 2],
                "target": [1, 2, 2, 3],
                "cost": [1.0, 4.0, 2.0, 1.0],
            }
        )
        dijkstra = Dijkstra(df_numeric, tail="source", head="target", weight="cost")
        path_lengths = dijkstra.run(0, return_inf=True)

        # Should produce valid results
        assert path_lengths[0] == 0.0
        assert len(path_lengths) == 4


@pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not available")
class TestPolarsAdvancedFeatures:
    """Test advanced Polars features and integration."""

    def test_polars_lazy_evaluation(self):
        """Test that lazy Polars operations work correctly."""
        # Create lazy frame
        df_lazy = pl.LazyFrame(
            {"tail": [0, 0, 1, 2], "head": [1, 2, 2, 3], "weight": [1.0, 4.0, 2.0, 1.0]}
        ).filter(
            pl.col("weight") < 5.0
        )  # Should include all rows

        # Collect to DataFrame
        df = df_lazy.collect()

        dijkstra = Dijkstra(df)
        path_lengths = dijkstra.run(0, return_inf=True)

        assert path_lengths[0] == 0.0
        assert path_lengths[1] == 1.0

    def test_polars_schema_preservation(self):
        """Test that Polars schema information is handled correctly."""
        # Create DataFrame with explicit schema
        schema = {"tail": pl.UInt32, "head": pl.UInt32, "weight": pl.Float64}

        df = pl.DataFrame(
            {
                "tail": [0, 0, 1, 2],
                "head": [1, 2, 2, 3],
                "weight": [1.0, 4.0, 2.0, 1.0],
            },
            schema=schema,
        )

        dijkstra = Dijkstra(df)

        # Conversion should preserve appropriate types
        assert dijkstra._edges["tail"].dtype == np.uint32
        assert dijkstra._edges["head"].dtype == np.uint32
        assert dijkstra._edges["weight"].dtype == np.float64

    def test_polars_null_handling(self):
        """Test handling of null values in Polars DataFrames."""
        df = pl.DataFrame(
            {
                "tail": [0, 0, None, 2],
                "head": [1, 2, 2, 3],
                "weight": [1.0, 4.0, 2.0, 1.0],
            }
        )

        # Should handle null values gracefully or raise appropriate error
        with pytest.raises((ValueError, TypeError)):
            Dijkstra(df, check_edges=True)

    def test_polars_hyperpath_support(self):
        """Test that Polars works with HyperpathGenerating algorithm."""
        from edsger.path import HyperpathGenerating

        df = pl.DataFrame(
            {
                "tail": [0, 0, 1, 2],
                "head": [1, 2, 2, 3],
                "trav_time": [1.0, 2.0, 1.0, 1.0],
                "freq": [0.1, 0.1, 0.1, 0.1],
            }
        )

        hp = HyperpathGenerating(df)
        hp.run(0, 3, 1.0, return_inf=True)

        # Should run without error and add volume column
        assert "volume" in hp._edges.columns

    def test_polars_large_dataset_conversion(self):
        """Test Polars conversion performance with larger datasets."""
        n = 10000
        np.random.seed(42)  # For reproducible results

        # Create large Polars DataFrame
        df = pl.DataFrame(
            {
                "tail": np.random.randint(0, 1000, n),
                "head": np.random.randint(0, 1000, n),
                "weight": np.random.random(n),
            }
        )

        # Should convert efficiently
        dijkstra = Dijkstra(df)

        # Internal representation should be efficient
        assert dijkstra._edges["tail"].dtype == np.uint32
        assert dijkstra._edges["head"].dtype == np.uint32
        # Length may be less than n due to duplicate edge removal
        assert len(dijkstra._edges) <= n
        assert len(dijkstra._edges) > 0  # Should have at least some edges

    def test_polars_dataframe_modifications(self):
        """Test that modifications to original Polars DataFrame don't affect algorithm."""
        df = pl.DataFrame(
            {"tail": [0, 0, 1, 2], "head": [1, 2, 2, 3], "weight": [1.0, 4.0, 2.0, 1.0]}
        )

        dijkstra = Dijkstra(df)

        # Modify original DataFrame
        df = df.with_columns(pl.col("weight") * 2)

        # Algorithm should use original data
        path_lengths = dijkstra.run(0, return_inf=True)
        assert path_lengths[1] == 1.0  # Not 2.0 (which would be from modified data)


@pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not available")
class TestPolarsErrorHandling:
    """Test error handling specific to Polars DataFrames."""

    def test_polars_invalid_column_names(self):
        """Test error handling for invalid column names."""
        df = pl.DataFrame(
            {"source": [0, 1, 2], "target": [1, 2, 3], "cost": [1.0, 2.0, 3.0]}
        )

        # Should raise error for non-existent columns
        # Polars raises ColumnNotFoundError which we should catch
        with pytest.raises(
            Exception
        ):  # Could be KeyError or polars.ColumnNotFoundError
            Dijkstra(df, tail="nonexistent_column")

    def test_polars_empty_dataframe(self):
        """Test handling of empty Polars DataFrames."""
        df = pl.DataFrame(
            {
                "tail": pl.Series([], dtype=pl.Int64),
                "head": pl.Series([], dtype=pl.Int64),
                "weight": pl.Series([], dtype=pl.Float64),
            }
        )

        dijkstra = Dijkstra(df)
        assert len(dijkstra._edges) == 0

    def test_polars_incompatible_dtypes(self):
        """Test handling of incompatible data types."""
        df = pl.DataFrame(
            {
                "tail": ["a", "b", "c"],  # String instead of numeric
                "head": [1, 2, 3],
                "weight": [1.0, 2.0, 3.0],
            }
        )

        # Should raise error during conversion or validation
        with pytest.raises((ValueError, TypeError)):
            Dijkstra(df, check_edges=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
