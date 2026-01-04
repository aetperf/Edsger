"""Cross-library DataFrame integration tests for Edsger graph algorithms."""

import pytest
import numpy as np
import pandas as pd
import time
from typing import Dict, Any

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

from edsger.path import Dijkstra, BellmanFord, HyperpathGenerating


class TestCrossLibraryConsistency:
    """Test that all DataFrame libraries produce identical results."""

    def create_test_data(self):
        """Create common test data."""
        return {
            "tail": [0, 0, 1, 1, 2, 2, 3],
            "head": [1, 2, 2, 3, 3, 4, 4],
            "weight": [1.0, 4.0, 2.0, 5.0, 1.0, 3.0, 2.0],
        }

    def create_dataframes(self, data: Dict[str, Any]):
        """Create DataFrames in all available formats."""
        dataframes = {}

        # NumPy-backed pandas
        dataframes["pandas_numpy"] = pd.DataFrame(data)

        # Arrow-backed pandas
        if PYARROW_AVAILABLE:
            df_arrow = pd.DataFrame(data)
            df_arrow = df_arrow.astype(
                {
                    "tail": pd.ArrowDtype(pa.int64()),
                    "head": pd.ArrowDtype(pa.int64()),
                    "weight": pd.ArrowDtype(pa.float64()),
                }
            )
            dataframes["pandas_arrow"] = df_arrow

        # Polars
        if POLARS_AVAILABLE:
            dataframes["polars"] = pl.DataFrame(data)

        return dataframes

    def test_dijkstra_cross_library_consistency(self):
        """Test that Dijkstra produces identical results across all DataFrame libraries."""
        data = self.create_test_data()
        dataframes = self.create_dataframes(data)

        results = {}
        for lib_name, df in dataframes.items():
            dijkstra = Dijkstra(df)
            results[lib_name] = dijkstra.run(0, return_inf=True)

        # All results should be identical
        reference_result = list(results.values())[0]
        for lib_name, result in results.items():
            np.testing.assert_array_equal(
                reference_result, result, err_msg=f"Results differ for {lib_name}"
            )

    def test_bellmanford_cross_library_consistency(self):
        """Test that BellmanFord produces identical results across all DataFrame libraries."""
        data = self.create_test_data()
        dataframes = self.create_dataframes(data)

        results = {}
        for lib_name, df in dataframes.items():
            bf = BellmanFord(df)
            results[lib_name] = bf.run(0, return_inf=True)

        # All results should be identical
        reference_result = list(results.values())[0]
        for lib_name, result in results.items():
            np.testing.assert_array_equal(
                reference_result, result, err_msg=f"Results differ for {lib_name}"
            )

    def test_hyperpath_cross_library_consistency(self):
        """Test that HyperpathGenerating produces consistent results."""
        data = {
            "tail": [0, 0, 1, 2],
            "head": [1, 2, 2, 3],
            "trav_time": [1.0, 2.0, 1.0, 1.0],
            "freq": [0.1, 0.1, 0.1, 0.1],
        }

        dataframes = {}
        dataframes["pandas_numpy"] = pd.DataFrame(data)

        if PYARROW_AVAILABLE:
            df_arrow = pd.DataFrame(data)
            df_arrow = df_arrow.astype(
                {
                    "tail": pd.ArrowDtype(pa.int64()),
                    "head": pd.ArrowDtype(pa.int64()),
                    "trav_time": pd.ArrowDtype(pa.float64()),
                    "freq": pd.ArrowDtype(pa.float64()),
                }
            )
            dataframes["pandas_arrow"] = df_arrow

        if POLARS_AVAILABLE:
            dataframes["polars"] = pl.DataFrame(data)

        # Run algorithms and compare final u_i_vec values
        u_i_vecs = {}
        for lib_name, df in dataframes.items():
            hp = HyperpathGenerating(df)
            hp.run(0, 3, 1.0, return_inf=True)
            u_i_vecs[lib_name] = hp.u_i_vec

        # All u_i_vec results should be identical
        reference_result = list(u_i_vecs.values())[0]
        for lib_name, result in u_i_vecs.items():
            np.testing.assert_array_equal(
                reference_result,
                result,
                err_msg=f"HyperpathGenerating u_i_vec differs for {lib_name}",
            )

    def test_path_tracking_consistency(self):
        """Test that path tracking produces identical results across libraries."""
        data = self.create_test_data()
        dataframes = self.create_dataframes(data)

        # Test path tracking with Dijkstra
        paths = {}
        for lib_name, df in dataframes.items():
            dijkstra = Dijkstra(df)
            dijkstra.run(0, path_tracking=True, return_inf=True)

            # Get path from source to target
            path = dijkstra.get_path(4)  # Path from 0 to 4
            paths[lib_name] = path

        # All paths should be identical
        reference_path = list(paths.values())[0]
        for lib_name, path in paths.items():
            if reference_path is None:
                assert path is None, f"Path differs for {lib_name}: expected None"
            else:
                np.testing.assert_array_equal(
                    reference_path, path, err_msg=f"Path differs for {lib_name}"
                )


class TestPerformanceComparison:
    """Compare performance across different DataFrame libraries."""

    def create_large_graph(self, n_vertices=1000, n_edges=5000):
        """Create a large random graph for performance testing."""
        np.random.seed(42)  # For reproducible results

        tail = np.random.randint(0, n_vertices, n_edges)
        head = np.random.randint(0, n_vertices, n_edges)
        weight = np.random.uniform(0.1, 10.0, n_edges)

        return {"tail": tail.tolist(), "head": head.tolist(), "weight": weight.tolist()}

    def measure_initialization_time(self, df, algorithm_class):
        """Measure time to initialize algorithm with given DataFrame."""
        start_time = time.time()
        _ = algorithm_class(df)
        end_time = time.time()
        return end_time - start_time

    def measure_run_time(self, algorithm):
        """Measure time to run algorithm."""
        start_time = time.time()
        algorithm.run(0, return_inf=True)
        end_time = time.time()
        return end_time - start_time

    @pytest.mark.performance
    def test_initialization_performance_comparison(self):
        """Compare initialization performance across DataFrame libraries."""
        data = self.create_large_graph(500, 2000)
        dataframes = {}

        # Create DataFrames
        dataframes["pandas_numpy"] = pd.DataFrame(data)

        if PYARROW_AVAILABLE:
            df_arrow = pd.DataFrame(data)
            df_arrow = df_arrow.astype(
                {
                    "tail": pd.ArrowDtype(pa.int64()),
                    "head": pd.ArrowDtype(pa.int64()),
                    "weight": pd.ArrowDtype(pa.float64()),
                }
            )
            dataframes["pandas_arrow"] = df_arrow

        if POLARS_AVAILABLE:
            dataframes["polars"] = pl.DataFrame(data)

        # Measure initialization times
        init_times = {}
        for lib_name, df in dataframes.items():
            init_time = self.measure_initialization_time(df, Dijkstra)
            init_times[lib_name] = init_time

        # Print timing results (for manual inspection)
        print("\nInitialization times:")
        for lib_name, time_taken in init_times.items():
            print(f"  {lib_name}: {time_taken:.4f} seconds")

        # All should complete in reasonable time (< 1 second for this size)
        for lib_name, time_taken in init_times.items():
            assert (
                time_taken < 1.0
            ), f"{lib_name} initialization too slow: {time_taken:.4f}s"

    @pytest.mark.performance
    def test_memory_usage_comparison(self):
        """Compare memory usage across DataFrame libraries."""
        data = self.create_large_graph(200, 1000)

        # Create algorithms
        algorithms = {}
        algorithms["pandas_numpy"] = Dijkstra(pd.DataFrame(data))

        if PYARROW_AVAILABLE:
            df_arrow = pd.DataFrame(data)
            df_arrow = df_arrow.astype(
                {
                    "tail": pd.ArrowDtype(pa.int64()),
                    "head": pd.ArrowDtype(pa.int64()),
                    "weight": pd.ArrowDtype(pa.float64()),
                }
            )
            algorithms["pandas_arrow"] = Dijkstra(df_arrow)

        if POLARS_AVAILABLE:
            algorithms["polars"] = Dijkstra(pl.DataFrame(data))

        # Check that all use consistent internal dtypes (optimal memory usage)
        for lib_name, algorithm in algorithms.items():
            # All should use integer types for vertex indices
            assert np.issubdtype(
                algorithm._edges["tail"].dtype, np.integer
            ), f"{lib_name} tail should be integer"
            assert np.issubdtype(
                algorithm._edges["head"].dtype, np.integer
            ), f"{lib_name} head should be integer"
            assert (
                algorithm._edges["weight"].dtype == np.float64
            ), f"{lib_name} weight dtype incorrect"

            # Check memory is contiguous
            assert algorithm._edges["tail"].values.flags[
                "C_CONTIGUOUS"
            ], f"{lib_name} tail not contiguous"
            assert algorithm._edges["head"].values.flags[
                "C_CONTIGUOUS"
            ], f"{lib_name} head not contiguous"
            assert algorithm._edges["weight"].values.flags[
                "C_CONTIGUOUS"
            ], f"{lib_name} weight not contiguous"


class TestRealWorldScenarios:
    """Test realistic scenarios with different DataFrame libraries."""

    def test_network_analysis_workflow(self):
        """Test a complete network analysis workflow."""
        # Create a realistic road network graph
        edges_data = {
            "from_intersection": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
            "to_intersection": [1, 2, 2, 3, 3, 4, 4, 5, 5, 0, 0],
            "travel_time": [5.0, 8.0, 3.0, 7.0, 2.0, 6.0, 4.0, 9.0, 1.0, 10.0, 12.0],
        }

        # Test with all available DataFrame libraries
        results = {}

        # pandas (NumPy backend)
        df_pandas = pd.DataFrame(edges_data)
        dijkstra = Dijkstra(
            df_pandas,
            tail="from_intersection",
            head="to_intersection",
            weight="travel_time",
        )
        results["pandas"] = dijkstra.run(0, return_inf=True)

        # pandas (Arrow backend)
        if PYARROW_AVAILABLE:
            df_arrow = pd.DataFrame(edges_data).astype(
                {
                    "from_intersection": pd.ArrowDtype(pa.int64()),
                    "to_intersection": pd.ArrowDtype(pa.int64()),
                    "travel_time": pd.ArrowDtype(pa.float64()),
                }
            )
            dijkstra_arrow = Dijkstra(
                df_arrow,
                tail="from_intersection",
                head="to_intersection",
                weight="travel_time",
            )
            results["pandas_arrow"] = dijkstra_arrow.run(0, return_inf=True)

        # Polars
        if POLARS_AVAILABLE:
            df_polars = pl.DataFrame(edges_data)
            dijkstra_polars = Dijkstra(
                df_polars,
                tail="from_intersection",
                head="to_intersection",
                weight="travel_time",
            )
            results["polars"] = dijkstra_polars.run(0, return_inf=True)

        # All results should be identical
        reference_result = list(results.values())[0]
        for lib_name, result in results.items():
            np.testing.assert_array_equal(reference_result, result)

    def test_transit_network_analysis(self):
        """Test transit network analysis with HyperpathGenerating."""
        # Create a transit network
        transit_data = {
            "from_stop": [0, 0, 1, 1, 2, 3],
            "to_stop": [1, 2, 2, 3, 3, 4],
            "travel_time": [2.0, 5.0, 1.5, 3.0, 2.5, 1.0],
            "frequency": [0.1, 0.05, 0.2, 0.15, 0.1, 0.3],
        }

        algorithms = {}

        # Test with pandas
        df_pandas = pd.DataFrame(transit_data)
        algorithms["pandas"] = HyperpathGenerating(
            df_pandas,
            tail="from_stop",
            head="to_stop",
            trav_time="travel_time",
            freq="frequency",
        )

        # Test with Arrow-backed pandas
        if PYARROW_AVAILABLE:
            df_arrow = pd.DataFrame(transit_data).astype(
                {
                    "from_stop": pd.ArrowDtype(pa.int64()),
                    "to_stop": pd.ArrowDtype(pa.int64()),
                    "travel_time": pd.ArrowDtype(pa.float64()),
                    "frequency": pd.ArrowDtype(pa.float64()),
                }
            )
            algorithms["pandas_arrow"] = HyperpathGenerating(
                df_arrow,
                tail="from_stop",
                head="to_stop",
                trav_time="travel_time",
                freq="frequency",
            )

        # Test with Polars
        if POLARS_AVAILABLE:
            df_polars = pl.DataFrame(transit_data)
            algorithms["polars"] = HyperpathGenerating(
                df_polars,
                tail="from_stop",
                head="to_stop",
                trav_time="travel_time",
                freq="frequency",
            )

        # Run algorithms and compare results
        results = {}
        for lib_name, algorithm in algorithms.items():
            algorithm.run(0, 4, 1.0, return_inf=True)
            results[lib_name] = algorithm.u_i_vec.copy()

        # Results should be consistent
        reference_result = list(results.values())[0]
        for lib_name, result in results.items():
            np.testing.assert_allclose(
                reference_result,
                result,
                rtol=1e-10,
                err_msg=f"Transit analysis differs for {lib_name}",
            )

    def test_mixed_data_types_handling(self):
        """Test handling of mixed data types across libraries."""
        # Data with different numeric types
        mixed_data = {
            "tail": [0, 1, 2, 3],  # Will be int64 by default
            "head": np.array([1, 2, 3, 4], dtype=np.int32),
            "weight": np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32),
        }

        # All libraries should handle this and normalize to standard types
        df_pandas = pd.DataFrame(mixed_data)
        dijkstra_pandas = Dijkstra(df_pandas)

        if POLARS_AVAILABLE:
            df_polars = pl.DataFrame(mixed_data)
            dijkstra_polars = Dijkstra(df_polars)

            # Results should be identical despite different input types
            result_pandas = dijkstra_pandas.run(0, return_inf=True)
            result_polars = dijkstra_polars.run(0, return_inf=True)
            np.testing.assert_array_equal(result_pandas, result_polars)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
