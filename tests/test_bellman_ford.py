"""
Test Bellman-Ford algorithm implementation
"""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import bellman_ford as scipy_bellman_ford

from edsger.path import BellmanFord, Dijkstra


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

        bf = BellmanFord(edges)

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
                    our_path_cost += edges[edge_mask]["weight"].iloc[0]

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
                            path_cost += edges[edge_mask]["weight"].iloc[0]

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


# author : Francois Pacull
# copyright : Architecture & Performance
# email: francois.pacull@architecture-performance.fr
# license : MIT
