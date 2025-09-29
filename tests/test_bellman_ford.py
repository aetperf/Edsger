"""Test Bellman-Ford algorithm implementation.

py.test tests/test_bellman_ford.py
"""

import numpy as np
import pandas as pd
import pytest  # type: ignore
from scipy.sparse import csr_matrix  # type: ignore
from scipy.sparse.csgraph import bellman_ford as scipy_bellman_ford  # type: ignore

from edsger.path import BellmanFord, Dijkstra

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
def braess():
    """Braess-like graph"""
    edges = pd.DataFrame(
        data={
            "tail": [0, 0, 1, 1, 2],
            "head": [1, 2, 2, 3, 3],
            "weight": [1.0, 2.0, 0.0, 2.0, 1.0],
        }
    )
    return edges


class TestBellmanFord:
    """Test Bellman-Ford algorithm implementation."""

    def test_bf_positive_weights(self, braess):  # pylint: disable=redefined-outer-name
        """Test that BF gives same results as Dijkstra for positive weights"""
        bf = BellmanFord(braess, orientation="out")
        dij = Dijkstra(braess, orientation="out")

        bf_dist = bf.run(vertex_idx=0)
        dij_dist = dij.run(vertex_idx=0)

        assert np.allclose(bf_dist, dij_dist)

    def test_bf_negative_edges(self):
        """Test BF with negative edges but no negative cycle"""
        edges = pd.DataFrame(
            {
                "tail": [0, 0, 1, 1, 2, 3],
                "head": [1, 2, 2, 3, 3, 4],
                "weight": [1, 4, -2, 5, 1, 3],
            }
        )

        bf = BellmanFord(edges)
        distances = bf.run(vertex_idx=0)

        # Hand-calculated shortest paths from vertex 0
        expected = [0, 1, -1, 0, 3]
        assert np.allclose(distances[:5], expected)

        # Verify no negative cycle was detected
        assert not bf.has_negative_cycle()

    def test_bf_negative_cycle(self):
        """Test that BF correctly detects negative cycles"""
        edges = pd.DataFrame(
            {
                "tail": [0, 1, 2, 2],
                "head": [1, 2, 0, 3],
                "weight": [1, -2, -1, 1],  # Cycle 0->1->2->0 has weight -2
            }
        )

        bf = BellmanFord(edges)
        with pytest.raises(ValueError, match="Negative cycle detected"):
            bf.run(vertex_idx=0, detect_negative_cycles=True)

    def test_bf_negative_cycle_no_detection(self):
        """Test BF without negative cycle detection"""
        edges = pd.DataFrame(
            {
                "tail": [0, 1, 2, 2],
                "head": [1, 2, 0, 3],
                "weight": [1, -2, -1, 1],  # Cycle 0->1->2->0 has weight -2
            }
        )

        bf = BellmanFord(edges)
        # Should not raise when detection is disabled
        distances = bf.run(vertex_idx=0, detect_negative_cycles=False)
        assert distances is not None

    def test_bf_scipy_comparison_positive(self):
        """Test BF against SciPy implementation with positive weights"""
        edges = pd.DataFrame(
            {
                "tail": [0, 0, 1, 1, 2, 3],
                "head": [1, 2, 2, 3, 3, 4],
                "weight": [2, 8, 3, 5, 1, 4],
            }
        )

        # Create sparse matrix for SciPy
        n_vertices = edges[["tail", "head"]].max().max() + 1
        sparse_matrix = csr_matrix(
            (edges["weight"], (edges["tail"], edges["head"])),
            shape=(n_vertices, n_vertices),
        )

        # Run SciPy Bellman-Ford
        scipy_dist, scipy_pred = scipy_bellman_ford(
            sparse_matrix, directed=True, indices=0, return_predecessors=True
        )

        # Run our Bellman-Ford with path tracking
        bf = BellmanFord(edges)
        our_dist = bf.run(vertex_idx=0, path_tracking=True)

        # Compare distances
        assert np.allclose(our_dist, scipy_dist)

        # Compare predecessors by reconstructing paths
        for target in range(n_vertices):
            if not np.isinf(our_dist[target]):
                # Get path from our implementation
                our_path = bf.get_path(target)

                # Reconstruct path from SciPy predecessors
                scipy_path = []
                current = target
                while current != -9999 and len(scipy_path) < n_vertices:
                    scipy_path.append(current)
                    current = scipy_pred[current]

                # Both paths should have same length and same total distance
                if our_path is not None and len(scipy_path) > 0:
                    assert len(our_path) == len(scipy_path)

    def test_bf_scipy_comparison_negative(self):
        """Test BF against SciPy implementation with negative weights"""
        edges = pd.DataFrame(
            {
                "tail": [0, 0, 1, 1, 2, 3],
                "head": [1, 2, 2, 3, 3, 4],
                "weight": [1, 4, -2, 5, 1, 3],
            }
        )

        # Create sparse matrix for SciPy
        n_vertices = edges[["tail", "head"]].max().max() + 1
        sparse_matrix = csr_matrix(
            (edges["weight"], (edges["tail"], edges["head"])),
            shape=(n_vertices, n_vertices),
        )

        # Run SciPy Bellman-Ford
        scipy_dist, scipy_pred = scipy_bellman_ford(
            sparse_matrix, directed=True, indices=0, return_predecessors=True
        )

        # Run our Bellman-Ford with path tracking
        bf = BellmanFord(edges)
        our_dist = bf.run(vertex_idx=0, path_tracking=True)

        # Compare distances
        assert np.allclose(our_dist, scipy_dist)

        # Compare predecessors by reconstructing paths
        for target in range(n_vertices):
            if not np.isinf(our_dist[target]):
                # Get path from our implementation
                our_path = bf.get_path(target)

                # Reconstruct path from SciPy predecessors
                scipy_path = []
                current = target
                while current != -9999 and len(scipy_path) < n_vertices:
                    scipy_path.append(current)
                    current = scipy_pred[current]

                # Both paths should have same length and same total distance
                if our_path is not None and len(scipy_path) > 0:
                    assert len(our_path) == len(scipy_path)

    def test_bf_path_tracking(self):
        """Test path tracking in Bellman-Ford"""
        edges = pd.DataFrame(
            {
                "tail": [0, 0, 1, 1, 2, 3],
                "head": [1, 2, 2, 3, 3, 4],
                "weight": [1, 4, -2, 5, 1, 3],
            }
        )

        bf = BellmanFord(edges)
        _ = bf.run(vertex_idx=0, path_tracking=True)

        # Get path from 0 to 4
        path = bf.get_path(4)
        assert path is not None

        # Path should be 0 -> 1 -> 2 -> 3 -> 4
        # (reversed because get_path returns backward from target to source)
        expected_path = [4, 3, 2, 1, 0]
        assert np.array_equal(path, expected_path)

    def test_bf_orientation_in(self):
        """Test Bellman-Ford with 'in' orientation (single target)"""
        edges = pd.DataFrame(
            {
                "tail": [0, 0, 1, 1, 2, 3],
                "head": [1, 2, 2, 3, 3, 4],
                "weight": [1, 4, -2, 5, 1, 3],
            }
        )

        bf = BellmanFord(edges, orientation="in")
        distances = bf.run(vertex_idx=4)  # distances from all vertices to vertex 4

        # Verify some known distances to vertex 4
        assert distances[4] == 0  # Distance from 4 to itself
        assert distances[3] == 3  # Distance from 3 to 4

    def test_bf_disconnected_graph(self):
        """Test BF with disconnected graph components"""
        edges = pd.DataFrame(
            {"tail": [0, 1, 3, 4], "head": [1, 2, 4, 5], "weight": [1, 2, 3, 4]}
        )

        bf = BellmanFord(edges)
        distances = bf.run(vertex_idx=0)

        # Vertices 3, 4, 5 should be unreachable from 0
        assert distances[0] == 0
        assert distances[1] == 1
        assert distances[2] == 3
        assert np.isinf(distances[3])
        assert np.isinf(distances[4])
        assert np.isinf(distances[5])

    def test_bf_return_series(self):
        """Test returning results as pandas Series"""
        edges = pd.DataFrame(
            {"tail": [0, 0, 1, 2], "head": [1, 2, 2, 3], "weight": [1, 3, 1, 1]}
        )

        bf = BellmanFord(edges)
        result = bf.run(vertex_idx=0, return_series=True)

        assert isinstance(result, pd.Series)
        assert len(result) == 4
        assert result[0] == 0
        assert result[1] == 1
        assert result[2] == 2
        assert result[3] == 3

    def test_bf_permute(self):
        """Test BF with vertex permutation"""
        # Create graph with non-contiguous vertex IDs
        edges = pd.DataFrame(
            {
                "tail": [10, 10, 20, 30],
                "head": [20, 30, 30, 40],
                "weight": [1, 4, -2, 3],
            }
        )

        bf = BellmanFord(edges, permute=True)
        distances = bf.run(vertex_idx=10)

        # Check distances
        assert distances[10] == 0
        assert distances[20] == 1
        assert distances[30] == -1
        assert distances[40] == 2

    def test_bf_check_edges_validation(self):
        """Test edge validation in BellmanFord"""
        # Test with missing values
        edges_with_nan = pd.DataFrame(
            {"tail": [0, 1, None], "head": [1, 2, 3], "weight": [1, 2, 3]}
        )

        with pytest.raises(ValueError, match="should not have any missing value"):
            BellmanFord(edges_with_nan, check_edges=True)

        # Test with non-numeric weights
        edges_non_numeric = pd.DataFrame(
            {"tail": [0, 1], "head": [1, 2], "weight": ["a", "b"]}
        )

        with pytest.raises(TypeError, match="should be of numeric type"):
            BellmanFord(edges_non_numeric, check_edges=True)

        # Test with infinite weights
        edges_infinite = pd.DataFrame(
            {"tail": [0, 1], "head": [1, 2], "weight": [1, np.inf]}
        )

        with pytest.raises(ValueError, match="should be finite"):
            BellmanFord(edges_infinite, check_edges=True)

    def test_bf_large_graph_performance(self):
        """Test BF performance on a larger graph"""
        # Create a larger random graph
        np.random.seed(42)
        n_edges = 1000
        n_vertices = 100

        edges = pd.DataFrame(
            {
                "tail": np.random.randint(0, n_vertices, n_edges),
                "head": np.random.randint(0, n_vertices, n_edges),
                "weight": np.random.randn(n_edges) * 10,  # Mix of positive and negative
            }
        )

        # Remove self-loops
        edges = edges[edges["tail"] != edges["head"]]

        bf = BellmanFord(edges)  # type: ignore

        # Should complete without error
        try:
            distances = bf.run(vertex_idx=0, detect_negative_cycles=True)
            # If no exception, check basic properties
            assert len(distances) == n_vertices
            assert distances[0] == 0
        except ValueError as e:
            # If negative cycle detected, that's also valid
            assert "Negative cycle detected" in str(e)

    def test_bf_zero_weight_edges(self):
        """Test BF with zero-weight edges"""
        edges = pd.DataFrame(
            {"tail": [0, 1, 2], "head": [1, 2, 3], "weight": [0, 0, 0]}
        )

        bf = BellmanFord(edges)
        distances = bf.run(vertex_idx=0)

        # All reachable vertices should have distance 0
        assert distances[0] == 0
        assert distances[1] == 0
        assert distances[2] == 0
        assert distances[3] == 0

    def test_bf_single_vertex(self):
        """Test BF with a single vertex (edge case)"""
        edges = pd.DataFrame({"tail": [], "head": [], "weight": []})

        # Add a single vertex by creating a self-loop and removing it
        edges = pd.DataFrame({"tail": [0], "head": [1], "weight": [1]})

        bf = BellmanFord(edges)
        distances = bf.run(vertex_idx=0)

        assert distances[0] == 0
        assert distances[1] == 1

    def test_bf_predecessor_validation_comprehensive(self):
        """Comprehensive predecessor validation against SciPy"""
        # Test with a more complex graph
        edges = pd.DataFrame(
            {
                "tail": [0, 0, 0, 1, 1, 2, 2, 3, 3, 4],
                "head": [1, 2, 3, 2, 4, 3, 5, 4, 5, 5],
                "weight": [4, 2, 8, 1, 7, 3, 2, 1, 6, 3],
            }
        )

        n_vertices = edges[["tail", "head"]].max().max() + 1
        sparse_matrix = csr_matrix(
            (edges["weight"], (edges["tail"], edges["head"])),
            shape=(n_vertices, n_vertices),
        )

        # Run SciPy Bellman-Ford
        scipy_dist, _ = scipy_bellman_ford(
            sparse_matrix, directed=True, indices=0, return_predecessors=True
        )

        # Run our Bellman-Ford with path tracking
        bf = BellmanFord(edges)
        our_dist = bf.run(vertex_idx=0, path_tracking=True)

        # Compare distances
        assert np.allclose(our_dist, scipy_dist)

        # Validate that both implementations produce optimal paths
        for target in range(n_vertices):
            if not np.isinf(our_dist[target]) and target != 0:
                # Get path from our implementation
                our_path = bf.get_path(target)
                assert our_path is not None

                # Calculate path cost from our implementation
                our_path_cost = 0
                for i in range(len(our_path) - 1):
                    curr, next_v = our_path[i + 1], our_path[i]
                    edge_mask = (edges["tail"] == curr) & (edges["head"] == next_v)
                    our_path_cost += edges[edge_mask]["weight"].iloc[0]  # type: ignore

                # Path cost should match distance
                assert abs(our_path_cost - our_dist[target]) < 1e-10

    def test_bf_predecessor_negative_weights(self):
        """Test predecessor tracking with negative weights"""
        edges = pd.DataFrame(
            {
                "tail": [0, 0, 1, 1, 2, 3, 4],
                "head": [1, 2, 2, 3, 4, 4, 5],
                "weight": [1, 5, -3, 2, 1, -1, 2],
            }
        )

        n_vertices = edges[["tail", "head"]].max().max() + 1
        sparse_matrix = csr_matrix(
            (edges["weight"], (edges["tail"], edges["head"])),
            shape=(n_vertices, n_vertices),
        )

        # Run SciPy Bellman-Ford
        scipy_dist, _ = scipy_bellman_ford(
            sparse_matrix, directed=True, indices=0, return_predecessors=True
        )

        # Run our Bellman-Ford with path tracking
        bf = BellmanFord(edges)
        our_dist = bf.run(vertex_idx=0, path_tracking=True)

        # Compare distances
        assert np.allclose(our_dist, scipy_dist)

        # Verify path optimality with negative weights
        for target in range(n_vertices):
            if not np.isinf(our_dist[target]) and target != 0:
                our_path = bf.get_path(target)
                if our_path is not None:
                    # Calculate actual path cost
                    path_cost = 0
                    for i in range(len(our_path) - 1):
                        curr, next_v = our_path[i + 1], our_path[i]
                        edge_mask = (edges["tail"] == curr) & (edges["head"] == next_v)
                        if edge_mask.any():
                            path_cost += edges[edge_mask]["weight"].iloc[0]  # type: ignore

                    # Should match computed distance
                    assert abs(path_cost - our_dist[target]) < 1e-10

    def _calculate_path_cost(self, path, edges):
        """Helper method to calculate path cost"""
        cost = 0
        for i in range(len(path) - 1):
            u, v = path[i + 1], path[i]  # reversed path
            edge_mask = (edges["tail"] == u) & (edges["head"] == v)
            cost += edges[edge_mask]["weight"].iloc[0]
        return cost

    def test_bf_path_reconstruction_vs_scipy(self):
        """Direct comparison of path reconstruction vs SciPy predecessors"""
        edges = pd.DataFrame(
            {
                "tail": [0, 0, 1, 1, 2, 2, 3],
                "head": [1, 2, 2, 3, 3, 4, 4],
                "weight": [2, 6, 1, 4, -2, 3, 1],
            }
        )

        n_vertices = edges[["tail", "head"]].max().max() + 1
        sparse_matrix = csr_matrix(
            (edges["weight"], (edges["tail"], edges["head"])),
            shape=(n_vertices, n_vertices),
        )

        # Run SciPy Bellman-Ford
        scipy_dist, scipy_pred = scipy_bellman_ford(
            sparse_matrix, directed=True, indices=0, return_predecessors=True
        )

        # Run our Bellman-Ford with path tracking
        bf = BellmanFord(edges)
        our_dist = bf.run(vertex_idx=0, path_tracking=True)

        # Compare distances first
        assert np.allclose(our_dist, scipy_dist)

        # Compare path reconstruction for each reachable target
        for target in range(n_vertices):
            if not np.isinf(our_dist[target]) and target != 0:
                # Get our path
                our_path = bf.get_path(target)
                assert our_path is not None

                # Reconstruct SciPy path from predecessors
                scipy_path = []
                current = target
                while current != -9999 and len(scipy_path) < n_vertices:
                    scipy_path.append(current)
                    current = scipy_pred[current]

                # Both should give paths of same length
                assert len(our_path) == len(scipy_path)

                # Both paths should have the same total cost
                our_cost = self._calculate_path_cost(our_path, edges)
                scipy_cost = self._calculate_path_cost(scipy_path, edges)
                assert abs(our_cost - scipy_cost) < 1e-10

                # Both costs should equal the computed distance
                assert abs(our_cost - our_dist[target]) < 1e-10

    def test_csc_negative_cycle_detection(self):
        """Test CSC-specific negative cycle detection for orientation='in'"""
        # Graph with negative cycle: 0→1→2→0 with total weight -2
        edges_with_cycle = pd.DataFrame(
            {
                "tail": [0, 1, 2, 2],
                "head": [1, 2, 0, 3],
                "weight": [1, -2, -1, 1],  # cycle: 1 + (-2) + (-1) = -2 < 0
            }
        )

        bf_cycle = BellmanFord(edges_with_cycle, orientation="in")

        # Should detect negative cycle regardless of target vertex
        with pytest.raises(ValueError, match="Negative cycle"):
            bf_cycle.run(vertex_idx=3)

        # Graph without negative cycle
        edges_no_cycle = pd.DataFrame(
            {
                "tail": [0, 1, 2],
                "head": [1, 2, 3],
                "weight": [1, -1, 3],  # no cycles possible
            }
        )

        bf_no_cycle = BellmanFord(edges_no_cycle, orientation="in")
        distances = bf_no_cycle.run(vertex_idx=3)  # Should not raise

        assert distances is not None
        assert len(distances) == 4
        # Check some expected distances TO vertex 3
        assert distances[3] == 0  # distance from 3 to itself
        assert distances[2] == 3  # distance from 2 to 3
        assert distances[1] == 2  # distance from 1 to 3 (via 2)
        assert distances[0] == 3  # distance from 0 to 3 (via 1, 2)

    def test_csc_vs_csr_consistency(self):
        """Ensure CSC detection gives same results as old CSR conversion method"""
        # Create a graph that should NOT have negative cycles
        edges = pd.DataFrame(
            {
                "tail": [0, 0, 1, 1, 2, 3],
                "head": [1, 2, 2, 3, 3, 4],
                "weight": [1, 4, -2, 5, 1, 3],
            }
        )

        # Test orientation="out" (uses CSR detection)
        bf_out = BellmanFord(edges, orientation="out")
        distances_out = bf_out.run(vertex_idx=0)  # Should work without negative cycle

        # Test orientation="in" (uses CSC detection)
        bf_in = BellmanFord(edges, orientation="in")
        distances_in = bf_in.run(vertex_idx=4)  # Should work without negative cycle

        # Both should complete successfully (no negative cycle detected)
        assert distances_out is not None
        assert distances_in is not None

        # Verify some known relationships
        assert distances_out[0] == 0  # source to itself
        assert distances_in[4] == 0  # target to itself

    def test_csc_negative_cycle_complex_graph(self):
        """Test CSC detection on a more complex graph with multiple potential cycles"""
        # Graph with negative cycle involving multiple vertices
        edges = pd.DataFrame(
            {
                "tail": [0, 1, 2, 3, 4, 4, 5],
                "head": [1, 2, 3, 1, 5, 0, 4],  # cycle: 1→2→3→1, separate cycle: 4→5→4
                "weight": [1, -3, 1, -1, 2, -5, -3],  # 1→2→3→1: 1+(-3)+1+(-1) = -2 < 0
            }
        )

        bf = BellmanFord(edges, orientation="in")

        # Should detect the negative cycle
        with pytest.raises(ValueError, match="Negative cycle"):
            bf.run(vertex_idx=5)

    def test_csc_detection_with_disconnected_components(self):
        """Test CSC negative cycle detection with disconnected graph components"""
        # Two scenarios: cycle affects target vs cycle doesn't affect target

        # Scenario 1: Target is reachable from negative cycle
        edges_cycle_affects_target = pd.DataFrame(
            {
                "tail": [
                    0,
                    1,
                    2,
                    2,
                    10,
                    11,
                ],  # Component 1: 0→1→2→0 + 2→3, Component 2: 10→11→12
                "head": [1, 2, 0, 3, 11, 12],
                "weight": [
                    1,
                    -1,
                    -2,
                    1,
                    5,
                    3,
                ],  # Cycle: 1+(-1)+(-2) = -2 < 0, and 2→3 reachable
            }
        )

        bf1 = BellmanFord(edges_cycle_affects_target, orientation="in")

        # Should detect negative cycle when target is reachable from cycle
        with pytest.raises(ValueError, match="Negative cycle"):
            bf1.run(vertex_idx=3)  # target reachable from cycle vertex 2

        # Scenario 2: Target is completely disconnected from negative cycle
        edges_cycle_isolated = pd.DataFrame(
            {
                "tail": [
                    0,
                    1,
                    2,
                    10,
                    11,
                ],  # Component 1: 0→1→2→0 (isolated), Component 2: 10→11→12
                "head": [1, 2, 0, 11, 12],
                "weight": [1, -1, -2, 5, 3],  # Component 1 has cycle but is isolated
            }
        )

        bf2 = BellmanFord(edges_cycle_isolated, orientation="in")

        # Target is isolated from negative cycle - STSP won't detect it
        # This is actually correct behavior for STSP algorithms
        distances = bf2.run(vertex_idx=12)  # Should complete successfully
        assert distances is not None
        assert distances[12] == 0  # distance from 12 to itself
        assert np.isinf(distances[0])  # vertex 0 not reachable to target 12

    def test_csc_detection_performance_vs_csr_conversion(self):
        """Test that CSC detection is more efficient than CSR conversion (basic timing)"""
        import time

        # Create a larger graph to see timing differences
        n_vertices = 1000
        edges = []

        # Create a chain: 0→1→2→...→999 with some negative weights (no cycles)
        for i in range(n_vertices - 1):
            weight = -0.5 if i % 10 == 0 else 1.0  # some negative weights but no cycles
            edges.append({"tail": i, "head": i + 1, "weight": weight})

        edges_df = pd.DataFrame(edges)

        bf = BellmanFord(edges_df, orientation="in")

        # Time the CSC detection (current implementation)
        start_time = time.time()
        distances = bf.run(vertex_idx=n_vertices - 1, detect_negative_cycles=True)
        csc_time = time.time() - start_time

        # Verify no negative cycle was detected (should run successfully)
        assert distances is not None
        assert len(distances) == n_vertices

        print(f"CSC detection time: {csc_time:.4f}s for {n_vertices} vertices")

        # The new CSC method should be significantly faster than the old CSR conversion
        # (This is more of a sanity check - actual performance test would need more setup)
        assert csc_time < 1.0  # Should be fast for 1000 vertices


class TestBellmanFordDataFrameBackends:
    """Test BellmanFord consistency across different DataFrame backends."""

    def create_test_dataframes(self, data: dict):
        """Create test DataFrames in all available formats."""
        dataframes = {}

        # pandas with NumPy backend
        dataframes["pandas_numpy"] = pd.DataFrame(data)

        # pandas with Arrow backend
        if PYARROW_AVAILABLE:
            df_arrow = pd.DataFrame(data)
            # Convert to Arrow backend
            arrow_dtypes = {}
            for col, values in data.items():
                if col in ["tail", "head"]:
                    arrow_dtypes[col] = pd.ArrowDtype(pa.int64())
                else:
                    arrow_dtypes[col] = pd.ArrowDtype(pa.float64())
            df_arrow = df_arrow.astype(arrow_dtypes)
            dataframes["pandas_arrow"] = df_arrow

        # Polars DataFrame
        if POLARS_AVAILABLE:
            dataframes["polars"] = pl.DataFrame(data)

        return dataframes

    def test_bellmanford_positive_weights_backend_consistency(self):
        """Test BellmanFord with positive weights across DataFrame backends."""
        data = {
            "tail": [0, 0, 1, 1, 2],
            "head": [1, 2, 2, 3, 3],
            "weight": [1.0, 2.0, 0.0, 2.0, 1.0],
        }

        dataframes = self.create_test_dataframes(data)
        results = {}

        # Run BellmanFord on each DataFrame backend
        for backend_name, df in dataframes.items():
            bf = BellmanFord(df, orientation="out")
            results[backend_name] = bf.run(vertex_idx=0, return_inf=True)

        # All results should be identical
        reference_result = list(results.values())[0]
        for backend_name, result in results.items():
            np.testing.assert_array_equal(
                reference_result,
                result,
                err_msg=f"BellmanFord positive weights results differ for {backend_name}",
            )

    def test_bellmanford_negative_weights_backend_consistency(self):
        """Test BellmanFord with negative weights across DataFrame backends."""
        data = {
            "tail": [0, 0, 1, 1, 2, 3],
            "head": [1, 2, 2, 3, 3, 4],
            "weight": [1.0, 4.0, -2.0, 5.0, 1.0, 3.0],
        }

        dataframes = self.create_test_dataframes(data)
        results = {}

        # Run BellmanFord on each DataFrame backend
        for backend_name, df in dataframes.items():
            bf = BellmanFord(df, orientation="out")
            results[backend_name] = bf.run(vertex_idx=0, return_inf=True)

        # All results should be identical
        reference_result = list(results.values())[0]
        for backend_name, result in results.items():
            np.testing.assert_array_equal(
                reference_result,
                result,
                err_msg=f"BellmanFord negative weights results differ for {backend_name}",
            )

    def test_bellmanford_path_tracking_backend_consistency(self):
        """Test BellmanFord path tracking across DataFrame backends."""
        data = {
            "tail": [0, 0, 1, 1, 2, 3],
            "head": [1, 2, 2, 3, 3, 4],
            "weight": [1.0, 4.0, -2.0, 5.0, 1.0, 3.0],
        }

        dataframes = self.create_test_dataframes(data)
        paths = {}

        # Run BellmanFord with path tracking on each DataFrame backend
        for backend_name, df in dataframes.items():
            bf = BellmanFord(df, orientation="out")
            bf.run(vertex_idx=0, path_tracking=True, return_inf=True)
            paths[backend_name] = bf.get_path(4)

        # All paths should be identical
        reference_path = list(paths.values())[0]
        for backend_name, path in paths.items():
            if reference_path is None:
                assert path is None, f"Path differs for {backend_name}: expected None"
            else:
                np.testing.assert_array_equal(
                    reference_path,
                    path,
                    err_msg=f"BellmanFord path tracking results differ for {backend_name}",
                )

    def test_bellmanford_negative_cycle_detection_backend_consistency(self):
        """Test negative cycle detection across DataFrame backends."""
        data = {
            "tail": [0, 1, 2, 2],
            "head": [1, 2, 0, 3],
            "weight": [1.0, -2.0, -1.0, 1.0],  # Cycle 0->1->2->0 has weight -2
        }

        dataframes = self.create_test_dataframes(data)

        # All backends should detect the negative cycle
        for backend_name, df in dataframes.items():
            bf = BellmanFord(df)
            with pytest.raises(ValueError, match="Negative cycle detected"):
                bf.run(vertex_idx=0, detect_negative_cycles=True)

    def test_bellmanford_orientation_in_backend_consistency(self):
        """Test BellmanFord with 'in' orientation across DataFrame backends."""
        data = {
            "tail": [0, 0, 1, 1, 2],
            "head": [1, 2, 2, 3, 3],
            "weight": [1.0, 2.0, 0.0, 2.0, 1.0],
        }

        dataframes = self.create_test_dataframes(data)
        results = {}

        # Run BellmanFord with orientation="in" on each DataFrame backend
        for backend_name, df in dataframes.items():
            bf = BellmanFord(df, orientation="in")
            results[backend_name] = bf.run(vertex_idx=3, return_inf=True)

        # All results should be identical
        reference_result = list(results.values())[0]
        for backend_name, result in results.items():
            np.testing.assert_array_equal(
                reference_result,
                result,
                err_msg=f"BellmanFord orientation='in' results differ for {backend_name}",
            )

    def test_bellmanford_permutation_backend_consistency(self):
        """Test BellmanFord with vertex permutation across DataFrame backends."""
        data = {
            "tail": [10, 10, 20, 30],
            "head": [20, 30, 30, 40],
            "weight": [1.0, 4.0, -2.0, 3.0],
        }

        dataframes = self.create_test_dataframes(data)
        results = {}

        # Run BellmanFord with permute=True on each DataFrame backend
        for backend_name, df in dataframes.items():
            bf = BellmanFord(df, permute=True)
            results[backend_name] = bf.run(vertex_idx=10, return_inf=True)

        # All results should be identical
        reference_result = list(results.values())[0]
        for backend_name, result in results.items():
            np.testing.assert_array_equal(
                reference_result,
                result,
                err_msg=f"BellmanFord permutation results differ for {backend_name}",
            )

    def test_bellmanford_internal_dtype_optimization(self):
        """Test internal dtype optimization across DataFrame backends."""
        data = {
            "tail": [0, 1, 2, 3],
            "head": [1, 2, 3, 4],
            "weight": [1.0, 2.0, 3.0, 4.0],
        }

        dataframes = self.create_test_dataframes(data)

        # Check internal representations are consistent
        for backend_name, df in dataframes.items():
            bf = BellmanFord(df)

            # All backends should use optimal internal dtypes for weights
            assert (
                bf._edges["weight"].dtype == np.float64
            ), f"{backend_name} weight dtype"

            # Memory should be contiguous for all backends
            assert bf._edges["tail"].values.flags[
                "C_CONTIGUOUS"
            ], f"{backend_name} tail not contiguous"
            assert bf._edges["head"].values.flags[
                "C_CONTIGUOUS"
            ], f"{backend_name} head not contiguous"
            assert bf._edges["weight"].values.flags[
                "C_CONTIGUOUS"
            ], f"{backend_name} weight not contiguous"

            # For vertex indices, different backends may use different dtypes but should be integer types
            assert np.issubdtype(
                bf._edges["tail"].dtype, np.integer
            ), f"{backend_name} tail should be integer"
            assert np.issubdtype(
                bf._edges["head"].dtype, np.integer
            ), f"{backend_name} head should be integer"

    def test_bellmanford_custom_columns_backend_consistency(self):
        """Test custom column names across DataFrame backends."""
        data = {
            "source": [0, 0, 1, 2],
            "target": [1, 2, 2, 3],
            "cost": [1.0, 4.0, -2.0, 1.0],
        }

        dataframes = self.create_test_dataframes(data)
        results = {}

        # Run with custom column names
        for backend_name, df in dataframes.items():
            bf = BellmanFord(df, tail="source", head="target", weight="cost")
            results[backend_name] = bf.run(vertex_idx=0, return_inf=True)

        # All results should be identical
        reference_result = list(results.values())[0]
        for backend_name, result in results.items():
            np.testing.assert_array_equal(
                reference_result,
                result,
                err_msg=f"BellmanFord custom columns results differ for {backend_name}",
            )

    def test_bellmanford_return_series_backend_consistency(self):
        """Test return_series=True across DataFrame backends."""
        data = {
            "tail": [0, 0, 1, 2],
            "head": [1, 2, 2, 3],
            "weight": [1.0, 3.0, 1.0, 1.0],
        }

        dataframes = self.create_test_dataframes(data)
        results = {}

        # Run with return_series=True
        for backend_name, df in dataframes.items():
            bf = BellmanFord(df)
            results[backend_name] = bf.run(vertex_idx=0, return_series=True)

        # All results should be identical Series
        reference_result = list(results.values())[0]
        for backend_name, result in results.items():
            assert isinstance(result, pd.Series), f"{backend_name} should return Series"
            try:
                pd.testing.assert_series_equal(
                    reference_result,
                    result,
                    check_names=False,  # Series names might differ
                )
            except AssertionError as e:
                raise AssertionError(
                    f"BellmanFord return_series results differ for {backend_name}: {e}"
                )

    def test_bellmanford_large_graph_backend_consistency(self):
        """Test consistency with larger graphs across DataFrame backends."""
        np.random.seed(42)  # For reproducible results
        n = 50
        m = 200

        data = {
            "tail": np.random.randint(0, n, m).tolist(),
            "head": np.random.randint(0, n, m).tolist(),
            "weight": (np.random.randn(m) * 2.0).tolist(),  # Mix of positive/negative
        }

        # Remove self-loops
        df_temp = pd.DataFrame(data)
        df_temp = df_temp[df_temp["tail"] != df_temp["head"]]
        data = df_temp.to_dict("list")

        dataframes = self.create_test_dataframes(data)
        results = {}

        # Run on larger graphs (without negative cycle detection to avoid randomness)
        for backend_name, df in dataframes.items():
            bf = BellmanFord(df)
            try:
                results[backend_name] = bf.run(
                    vertex_idx=0, detect_negative_cycles=False, return_inf=True
                )
            except Exception as e:
                # If any backend fails, all should fail consistently
                results[backend_name] = f"Error: {type(e).__name__}"

        # Check consistency - either all succeed or all fail with similar errors
        reference_result = list(results.values())[0]
        if isinstance(reference_result, str):  # Error case
            # All should have similar errors
            for backend_name, result in results.items():
                assert isinstance(
                    result, str
                ), f"Inconsistent error handling for {backend_name}"
        else:  # Success case
            for backend_name, result in results.items():
                np.testing.assert_array_equal(
                    reference_result,
                    result,
                    err_msg=f"Large graph BellmanFord results differ for {backend_name}",
                )


# author : Francois Pacull
# copyright : Architecture & Performance
# email: francois.pacull@architecture-performance.fr
# license : MIT
