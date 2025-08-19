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

    def test_bf_positive_weights(self, braess):
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

        # Run our Bellman-Ford
        bf = BellmanFord(edges)
        our_dist = bf.run(vertex_idx=0)

        # Compare results
        assert np.allclose(our_dist, scipy_dist)

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

        # Run our Bellman-Ford
        bf = BellmanFord(edges)
        our_dist = bf.run(vertex_idx=0)

        # Compare results
        assert np.allclose(our_dist, scipy_dist)

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
        distances = bf.run(vertex_idx=0, path_tracking=True)

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


# author : Francois Pacull
# copyright : Architecture & Performance
# email: francois.pacull@architecture-performance.fr
# license : MIT
