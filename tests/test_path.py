"""Tests for path.py.

py.test tests/test_path.py
"""

import numpy as np
import pandas as pd
import pytest  # type: ignore
from edsger.commons import DTYPE_INF_PY  # A_VERY_SMALL_TIME_INTERVAL_PY unused
from edsger.networks import create_sf_network
from edsger.path import BellmanFord, BFS, Dijkstra, HyperpathGenerating
from scipy.sparse import coo_array  # type: ignore  # csr_matrix unused
from scipy.sparse.csgraph import breadth_first_order, dijkstra  # type: ignore

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


@pytest.fixture
def spiess_florian_network():
    edges = create_sf_network()
    return edges


def test_check_edges_01():
    """Negative weights."""

    edges = pd.DataFrame(
        data={
            "tail": [0, 0],
            "head": [1, 2],
            "weight": [1.0, -2.0],
        }
    )
    with pytest.raises(ValueError, match=r"nonnegative"):
        _ = Dijkstra(
            edges,
            check_edges=True,
        )


def test_check_edges_02(braess):
    edges = braess

    with pytest.raises(TypeError, match=r"pandas DataFrame"):
        _ = Dijkstra("yeaaahhh!!!", check_edges=True)  # type: ignore
    with pytest.raises(KeyError, match=r"not found in graph edges dataframe"):
        _ = Dijkstra(edges, tail="source", check_edges=True)
    with pytest.raises(KeyError, match=r"not found in graph edges dataframe"):
        _ = Dijkstra(edges, head="target", check_edges=True)
    with pytest.raises(KeyError, match=r"not found in graph edges dataframe"):
        _ = Dijkstra(edges, weight="cost", check_edges=True)
    with pytest.raises(ValueError, match=r"missing value"):
        _ = Dijkstra(edges.replace(0, np.nan), check_edges=True)
    with pytest.raises(TypeError, match=r"should be of integer type"):
        _ = Dijkstra(edges.astype({"tail": float}), check_edges=True)
    with pytest.raises(TypeError, match=r"should be of numeric type"):
        _ = Dijkstra(edges.astype({"weight": str}), check_edges=True)


def test_run_01(braess):
    edges = braess
    sp = Dijkstra(edges, orientation="out", check_edges=False)
    path_lengths = sp.run(vertex_idx=0, return_series=True)
    path_lengths_ref = pd.Series([0.0, 1.0, 1.0, 2.0])
    path_lengths_ref.index.name = "vertex_idx"
    path_lengths_ref.name = "path_length"
    pd.testing.assert_series_equal(path_lengths, path_lengths_ref)


def test_run_02(random_seed=124, n=1000):
    np.random.seed(random_seed)
    tail = np.random.randint(0, int(n / 5), n)
    head = np.random.randint(0, int(n / 5), n)
    weight = np.random.rand(n)
    edges = pd.DataFrame(data={"tail": tail, "head": head, "weight": weight})
    edges.drop_duplicates(subset=["tail", "head"], inplace=True)
    edges = edges.loc[edges["tail"] != edges["head"]]
    edges.reset_index(drop=True, inplace=True)

    # SciPy
    vertex_count = edges[["tail", "head"]].max().max() + 1
    data = edges["weight"].values
    row = edges["tail"].values.astype(np.int32)
    col = edges["head"].values.astype(np.int32)
    graph_coo = coo_array((data, (row, col)), shape=(vertex_count, vertex_count))
    graph_csr = graph_coo.tocsr()
    dist_matrix_ref = dijkstra(
        csgraph=graph_csr, directed=True, indices=0, return_predecessors=False
    )

    # In-house
    # without graph permutation
    # return_inf=True
    sp = Dijkstra(edges, orientation="out", check_edges=True, permute=False)
    dist_matrix = sp.run(vertex_idx=0, return_inf=True)
    assert np.allclose(dist_matrix, dist_matrix_ref)

    dist_matrix_ref = np.where(
        dist_matrix_ref > DTYPE_INF_PY, DTYPE_INF_PY, dist_matrix_ref
    )

    # without graph permutation
    # return_inf=False
    dist_matrix = sp.run(vertex_idx=0, return_inf=False)
    assert np.allclose(dist_matrix, dist_matrix_ref)

    # with graph permutation
    # return_inf=False
    sp = Dijkstra(edges, orientation="out", check_edges=True, permute=True)
    dist_matrix = sp.run(vertex_idx=0, return_inf=False)
    assert np.allclose(dist_matrix, dist_matrix_ref)


def test_run_03(braess):
    """
    orientation="in"
    """
    edges = braess
    sp = Dijkstra(edges, orientation="in")
    path_lengths = sp.run(vertex_idx=3)
    path_lengths_ref = [2.0, 1.0, 1.0, 0.0]
    assert np.allclose(path_lengths, path_lengths_ref)


def test_run_04():
    """
    permute=True
    """
    edges = pd.DataFrame(
        data={
            "tail": [0, 0, 10, 10, 20],
            "head": [10, 20, 20, 30, 30],
            "weight": [1.0, 2.0, 0.0, 2.0, 1.0],
        }
    )
    sp = Dijkstra(edges, orientation="out", permute=True)
    path_lengths = sp.run(
        vertex_idx=0, path_tracking=True, return_inf=True, return_series=False
    )
    path_lengths_ref = np.array(
        [
            0.0,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            1.0,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            1.0,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            2.0,
        ]
    )
    assert np.allclose(path_lengths, path_lengths_ref)
    path_links_ref = np.array(
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            0,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            10,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            20,
        ],
        dtype=np.uint32,
    )
    np.testing.assert_array_equal(sp._path_links, path_links_ref)

    path_lengths = sp.run(
        vertex_idx=0, path_tracking=True, return_inf=False, return_series=False
    )
    path_lengths_ref = np.array(
        [
            0.00000000e000,
            DTYPE_INF_PY,
            DTYPE_INF_PY,
            DTYPE_INF_PY,
            DTYPE_INF_PY,
            DTYPE_INF_PY,
            DTYPE_INF_PY,
            DTYPE_INF_PY,
            DTYPE_INF_PY,
            DTYPE_INF_PY,
            1.00000000e000,
            DTYPE_INF_PY,
            DTYPE_INF_PY,
            DTYPE_INF_PY,
            DTYPE_INF_PY,
            DTYPE_INF_PY,
            DTYPE_INF_PY,
            DTYPE_INF_PY,
            DTYPE_INF_PY,
            DTYPE_INF_PY,
            1.00000000e000,
            DTYPE_INF_PY,
            DTYPE_INF_PY,
            DTYPE_INF_PY,
            DTYPE_INF_PY,
            DTYPE_INF_PY,
            DTYPE_INF_PY,
            DTYPE_INF_PY,
            DTYPE_INF_PY,
            DTYPE_INF_PY,
            2.00000000e000,
        ]
    )
    assert np.allclose(path_lengths, path_lengths_ref)
    np.testing.assert_array_equal(sp._path_links, path_links_ref)

    path_lengths = sp.run(vertex_idx=0, path_tracking=True, return_series=True)
    path_lengths_ref = pd.Series([0.0, 1.0, 1.0, 2.0], index=[0, 10, 20, 30])
    path_lengths_ref.index.name = "vertex_idx"
    path_lengths_ref.name = "path_length"
    pd.testing.assert_series_equal(path_lengths, path_lengths_ref)
    path_links_ref = pd.Series([0, 0, 10, 20], index=[0, 10, 20, 30])
    path_links_ref.index.name = "vertex_idx"
    path_links_ref.name = "associated_idx"
    path_links_ref = path_links_ref.astype(np.uint32)
    np.testing.assert_array_equal(sp._path_links, path_links_ref)


def test_path_tracking_01():
    edges = pd.DataFrame(
        data={
            "tail": [0, 0, 10, 10, 20],
            "head": [10, 20, 20, 30, 30],
            "weight": [1.0, 2.0, 0.0, 2.0, 1.0],
        }
    )
    sp = Dijkstra(edges, orientation="out", permute=True)

    # trying to get the path while no shortest path alrgorithm
    # has been executed yet
    with pytest.warns(UserWarning) as record:
        sp.get_path(0)

    # check that only one warning was raised
    assert len(record) == 1

    # run the shortest path algorithm
    _ = sp.run(
        vertex_idx=0, path_tracking=True, return_inf=True, return_series=False
    )  # path_lengths unused
    path_links_ref = np.array(
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            0,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            10,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            20,
        ],
        dtype=np.uint32,
    )
    np.testing.assert_array_equal(sp._path_links, path_links_ref)

    path_vertices = sp.get_path(30)
    path_vertices_ref = np.array([30, 20, 10, 0], dtype=np.uint32)
    np.testing.assert_array_equal(path_vertices, path_vertices_ref)


def test_SF_in_01():
    edges = create_sf_network(dwell_time=0.0)
    hp = HyperpathGenerating(edges, check_edges=False)
    hp.run(origin=0, destination=12, volume=1.0)

    assert np.allclose(edges["volume_ref"].values, hp._edges["volume"].values)  # type: ignore

    u_i_vec_ref = np.array(
        [
            1.66500000e03,
            1.47000000e03,
            1.50000000e03,
            1.14428572e03,
            4.80000000e02,
            1.05000000e03,
            1.05000000e03,
            6.90000000e02,
            6.00000000e02,
            2.40000000e02,
            2.40000000e02,
            6.90000000e02,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
        ]
    )
    assert np.allclose(u_i_vec_ref, hp.u_i_vec, rtol=1e-08, atol=1e-08)  # type: ignore


def test_SF_dwell_and_transfer_01():
    """
    One line, one stop case. We look at what is happening at a given stop.
    Do people use the dwell edge (what they should do) or the alighting and
    boarding edges.

            0
      0 --------> 2
        \\      ^
        1\\    /2
          \\  /
           V /
            1

    Network with 3 vertices and 3 edges.
    """

    # line 1
    line_freq = 1.0 / 600.0
    dwell_time = 5.0

    # stop A
    # 0 alighting vertex
    # 1 stop vertex
    # 2 boarding vertex

    tail = []
    head = []
    trav_time = []
    freq = []
    vol = []

    # edge 0
    # stop A : in line 1
    # dwell edge
    tail.append(0)
    head.append(2)
    freq.append(np.inf)
    trav_time.append(dwell_time)
    vol.append(1.0)

    # edge 1
    # stop A : from line 1
    # alighting edge
    tail.append(0)
    head.append(1)
    freq.append(np.inf)
    trav_time.append(0.5 * dwell_time)
    vol.append(0.0)

    # edge 2
    # stop A : to line 1
    # boarding edge
    tail.append(1)
    head.append(2)
    freq.append(line_freq)
    trav_time.append(0.5 * dwell_time)
    vol.append(0.0)

    edges = pd.DataFrame(
        data={
            "tail": tail,
            "head": head,
            "trav_time": trav_time,
            "freq": freq,
            "volume_ref": vol,
        }
    )
    # waiting time is in average half of the period
    edges["freq"] *= 2.0

    # SF
    hp = HyperpathGenerating(edges, check_edges=False)
    hp.run(origin=0, destination=2, volume=1.0)

    assert np.allclose(edges["volume_ref"].values, hp._edges["volume"].values)  # type: ignore

    u_i_vec_ref = np.array(
        [
            dwell_time,
            0.5 * (1.0 / line_freq + dwell_time),
            0.0,
        ]
    )
    assert np.allclose(u_i_vec_ref, hp.u_i_vec, rtol=1e-08, atol=1e-08)  # type: ignore

    # now we change the dwell edge into a transfer edge
    freq[0] = line_freq

    # vol = 3*[0.5]
    edges = pd.DataFrame(
        data={
            "tail": tail,
            "head": head,
            "trav_time": trav_time,
            "freq": freq,
            "volume_ref": vol,
        }
    )
    # waiting time is in average half of the period
    edges["freq"] *= 2.0

    # SF
    hp = HyperpathGenerating(edges, check_edges=False)
    hp.run(origin=0, destination=2, volume=1.0)

    # edges 1 and 2 are not used because edge 1 does not have
    # a real infinite frequency but INF_FREQ value, for numerical reasons.
    # This implies a small resistance to the path going through vertex 1.
    # If we decrease by a tiny amount the frequency of edge 0, the flow goes
    # though vertex 1.
    assert np.allclose(edges["volume_ref"].values, hp._edges["volume"].values)  # type: ignore

    u_i_vec_ref = [305.0, 302.5, 0.0]
    assert np.allclose(u_i_vec_ref, hp.u_i_vec, rtol=1e-08, atol=1e-08)  # type: ignore


def test_SF_network_run_01(spiess_florian_network):
    """
    Test from Spiess, H. and Florian, M. (1989).
    Optimal strategies: A new assignment model for transit networks.
    Transportation Research Part B 23(2), 83-102.

    This test validates both edge volumes and vertex travel times (u_i_vec)
    against the reference values from the paper.
    """
    edges = spiess_florian_network

    hp = HyperpathGenerating(edges)
    hp.run(origin=0, destination=12, volume=1.0)

    # Test edge volumes - merge by (tail, head) to ensure correct alignment
    # since HyperpathGenerating may reorder edges internally
    edges_with_ref = edges[["tail", "head", "volume_ref"]].copy()
    computed_volumes = hp._edges[["tail", "head", "volume"]].copy()
    merged = edges_with_ref.merge(computed_volumes, on=["tail", "head"])

    # Verify total volume conservation (should sum to 1.0 at origin edges)
    origin_edges = merged[merged["tail"] == 0]
    assert np.isclose(origin_edges["volume"].sum(), 1.0, rtol=1e-05)

    # Compare volumes with tolerance for floating point artifacts
    np.testing.assert_allclose(
        merged["volume"].values, merged["volume_ref"].values, rtol=1e-05, atol=1e-06
    )

    # Test vertex travel times (u_i_vec) from the paper
    # These are the expected travel times from each vertex to the destination (vertex 12)
    u_i_vec_ref = np.array(
        [
            1.66500000e03,  # vertex 0 (stop A)
            1.47000000e03,  # vertex 1
            1.50000000e03,  # vertex 2
            1.14428572e03,  # vertex 3 (stop X)
            4.80000000e02,  # vertex 4
            1.05000000e03,  # vertex 5
            1.05000000e03,  # vertex 6
            6.90000000e02,  # vertex 7 (stop Y)
            6.00000000e02,  # vertex 8
            2.40000000e02,  # vertex 9
            2.40000000e02,  # vertex 10
            6.90000000e02,  # vertex 11
            0.00000000e00,  # vertex 12 (stop B - destination)
            0.00000000e00,  # vertex 13
            0.00000000e00,  # vertex 14
            0.00000000e00,  # vertex 15
        ]
    )

    # Use atol=1e-06 to account for small dwell_time effects at destination vertices
    np.testing.assert_allclose(u_i_vec_ref, hp.u_i_vec, rtol=1e-05, atol=1e-06)


def test_dijkstra_early_termination_sssp(braess):
    """Test SSSP early termination functionality."""

    # test with orientation="out" (SSSP)
    dij = Dijkstra(braess, orientation="out")

    # run full algorithm to get reference
    path_lengths_ref = dij.run(vertex_idx=0)

    # run with early termination - stop when vertices 1 and 3 are reached
    termination_nodes = [1, 3]
    path_lengths = dij.run(vertex_idx=0, termination_nodes=termination_nodes)

    # early termination returns only distances to termination nodes
    expected_distances = [
        path_lengths_ref[1],
        path_lengths_ref[3],
    ]  # distances to nodes 1 and 3
    assert np.allclose(path_lengths, expected_distances)

    # test with path tracking
    path_lengths_tracked = dij.run(
        vertex_idx=0, termination_nodes=termination_nodes, path_tracking=True
    )
    assert np.allclose(path_lengths_tracked, expected_distances)


def test_dijkstra_early_termination_stsp(braess):
    """Test STSP early termination functionality."""

    # test with orientation="in" (STSP)
    dij = Dijkstra(braess, orientation="in")

    # run full algorithm to get reference
    path_lengths_ref = dij.run(vertex_idx=3)

    # run with early termination - stop when vertices 0 and 2 are reached
    termination_nodes = [0, 2]
    path_lengths = dij.run(vertex_idx=3, termination_nodes=termination_nodes)

    # early termination returns only distances from termination nodes
    expected_distances = [
        path_lengths_ref[0],
        path_lengths_ref[2],
    ]  # distances from nodes 0 and 2
    assert np.allclose(path_lengths, expected_distances)

    # test with path tracking
    path_lengths_tracked = dij.run(
        vertex_idx=3, termination_nodes=termination_nodes, path_tracking=True
    )
    assert np.allclose(path_lengths_tracked, expected_distances)


def test_dijkstra_early_termination_sssp_only(braess):
    """Test SSSP early termination without path tracking specifically."""

    # test with orientation="out" (SSSP) without path tracking
    dij = Dijkstra(braess, orientation="out")

    # run full algorithm first to get reference
    path_lengths_full = dij.run(vertex_idx=0, path_tracking=False)

    # run with early termination - target specific nodes
    termination_nodes = [2, 3]
    path_lengths_early = dij.run(
        vertex_idx=0, termination_nodes=termination_nodes, path_tracking=False
    )

    # verify correctness: early termination should return distances to termination nodes only
    expected_distances = [
        path_lengths_full[2],
        path_lengths_full[3],
    ]  # distances to nodes 2 and 3
    assert np.allclose(path_lengths_early, expected_distances)

    # test with single termination node
    path_lengths_single = dij.run(
        vertex_idx=0, termination_nodes=[1], path_tracking=False
    )
    expected_single = [path_lengths_full[1]]  # distance to node 1
    assert np.allclose(path_lengths_single, expected_single)


def test_dijkstra_early_termination_stsp_with_paths(braess):
    """Test STSP early termination with path tracking specifically."""

    # test with orientation="in" (STSP) with path tracking
    dij = Dijkstra(braess, orientation="in")

    # run full algorithm first to get reference
    path_lengths_full = dij.run(vertex_idx=3, path_tracking=True)

    # run with early termination and path tracking
    termination_nodes = [1, 2]
    path_lengths_early = dij.run(
        vertex_idx=3, termination_nodes=termination_nodes, path_tracking=True
    )

    # verify correctness: early termination should return distances from termination nodes only
    expected_distances = [
        path_lengths_full[1],
        path_lengths_full[2],
    ]  # distances from nodes 1 and 2
    assert np.allclose(path_lengths_early, expected_distances)

    # verify path tracking works
    assert dij.path_links is not None

    # test path extraction for a node that was computed
    # Note: for STSP, paths are from termination nodes to target
    path_to_1 = dij.get_path(1)
    assert path_to_1 is not None


def test_dijkstra_early_termination_sssp_permute_true(braess):
    """Test SSSP early termination functionality with permute=True."""

    # test with orientation="out" (SSSP) and permute=True
    dij = Dijkstra(braess, orientation="out", permute=True)

    # run full algorithm to get reference
    path_lengths_ref = dij.run(vertex_idx=0)

    # run with early termination - stop when vertices 1 and 3 are reached
    termination_nodes = [1, 3]
    path_lengths = dij.run(vertex_idx=0, termination_nodes=termination_nodes)

    # early termination returns only distances to termination nodes
    expected_distances = [
        path_lengths_ref[1],
        path_lengths_ref[3],
    ]  # distances to nodes 1 and 3
    assert np.allclose(path_lengths, expected_distances)

    # test with single termination node
    path_lengths_single = dij.run(vertex_idx=0, termination_nodes=[2])
    expected_single = [path_lengths_ref[2]]  # distance to node 2
    assert np.allclose(path_lengths_single, expected_single)

    # verify that results with permute=True match permute=False results
    dij_no_permute = Dijkstra(braess, orientation="out", permute=False)
    path_lengths_no_permute = dij_no_permute.run(
        vertex_idx=0, termination_nodes=termination_nodes
    )
    assert np.allclose(path_lengths, path_lengths_no_permute)


def test_dijkstra_early_termination_stsp_permute_true(braess):
    """Test STSP early termination functionality with permute=True."""

    # test with orientation="in" (STSP) and permute=True
    dij = Dijkstra(braess, orientation="in", permute=True)

    # run full algorithm to get reference
    path_lengths_ref = dij.run(vertex_idx=3)

    # run with early termination - stop when vertices 0 and 2 are reached
    termination_nodes = [0, 2]
    path_lengths = dij.run(vertex_idx=3, termination_nodes=termination_nodes)

    # early termination returns only distances from termination nodes
    expected_distances = [
        path_lengths_ref[0],
        path_lengths_ref[2],
    ]  # distances from nodes 0 and 2
    assert np.allclose(path_lengths, expected_distances)

    # test with single termination node
    path_lengths_single = dij.run(vertex_idx=3, termination_nodes=[1])
    expected_single = [path_lengths_ref[1]]  # distance from node 1
    assert np.allclose(path_lengths_single, expected_single)

    # verify that results with permute=True match permute=False results
    dij_no_permute = Dijkstra(braess, orientation="in", permute=False)
    path_lengths_no_permute = dij_no_permute.run(
        vertex_idx=3, termination_nodes=termination_nodes
    )
    assert np.allclose(path_lengths, path_lengths_no_permute)


# ============================================================================ #
# Bellman-Ford tests                                                          #
# ============================================================================ #


def test_bellman_ford_positive_weights(braess):
    """Test BellmanFord with positive weights matches Dijkstra."""

    # Run Dijkstra
    dij = Dijkstra(braess, orientation="out")
    dij_dist = dij.run(vertex_idx=0)

    # Run Bellman-Ford
    bf = BellmanFord(braess, orientation="out")
    bf_dist = bf.run(vertex_idx=0)

    # Results should match for positive weights
    assert np.allclose(bf_dist, dij_dist)


def test_bellman_ford_negative_edges():
    """Test BellmanFord with negative edges."""

    edges = pd.DataFrame(
        data={
            "tail": [0, 0, 1, 1, 2, 3],
            "head": [1, 2, 2, 3, 3, 4],
            "weight": [1.0, 4.0, -2.0, 5.0, 1.0, 3.0],
        }
    )

    bf = BellmanFord(edges)
    distances = bf.run(vertex_idx=0)

    # Expected shortest paths from vertex 0
    expected = np.array([0.0, 1.0, -1.0, 0.0, 3.0])
    assert np.allclose(distances[:5], expected)

    # Verify no negative cycle
    assert not bf.has_negative_cycle()


def test_bellman_ford_negative_cycle_detection():
    """Test BellmanFord negative cycle detection."""

    edges = pd.DataFrame(
        data={
            "tail": [0, 1, 2, 2],
            "head": [1, 2, 0, 3],
            "weight": [1.0, -2.0, -1.0, 1.0],  # Cycle 0->1->2->0 has weight -2
        }
    )

    bf = BellmanFord(edges)

    # Should raise ValueError when negative cycle detected
    with pytest.raises(ValueError, match="Negative cycle detected"):
        bf.run(vertex_idx=0, detect_negative_cycles=True)


def test_bellman_ford_path_tracking():
    """Test BellmanFord path tracking."""

    edges = pd.DataFrame(
        data={
            "tail": [0, 0, 1, 1, 2, 3],
            "head": [1, 2, 2, 3, 3, 4],
            "weight": [1.0, 4.0, -2.0, 5.0, 1.0, 3.0],
        }
    )

    bf = BellmanFord(edges)
    _ = bf.run(vertex_idx=0, path_tracking=True)  # distances unused

    # Get path from 0 to 4
    path = bf.get_path(4)
    assert path is not None

    # Verify the path is correct (backward from target to source)
    assert path[0] == 4  # starts at target
    assert path[-1] == 0  # ends at source


def test_bellman_ford_orientation_in():
    """Test BellmanFord with 'in' orientation."""

    edges = pd.DataFrame(
        data={
            "tail": [0, 0, 1, 1, 2],
            "head": [1, 2, 2, 3, 3],
            "weight": [1.0, 2.0, 0.0, 2.0, 1.0],
        }
    )

    bf_out = BellmanFord(edges, orientation="out")
    bf_in = BellmanFord(edges, orientation="in")

    # SSSP from vertex 0
    _ = bf_out.run(vertex_idx=0)  # dist_from_0 unused

    # STSP to vertex 0
    dist_to_0 = bf_in.run(vertex_idx=0)

    # dist_to_0[i] should be the distance from vertex i to vertex 0
    # This is equivalent to the distance from 0 to i in the reverse graph
    assert dist_to_0[0] == 0  # Distance from 0 to itself


def test_bellman_ford_permute():
    """Test BellmanFord with vertex permutation."""

    # Use non-contiguous vertex IDs
    edges = pd.DataFrame(
        data={
            "tail": [10, 10, 20, 30],
            "head": [20, 30, 30, 40],
            "weight": [1.0, 4.0, -2.0, 3.0],
        }
    )

    bf = BellmanFord(edges, permute=True)
    distances = bf.run(vertex_idx=10)

    # Check distances
    assert distances[10] == 0.0
    assert distances[20] == 1.0
    assert distances[30] == -1.0
    assert distances[40] == 2.0


def test_bellman_ford_return_series():
    """Test BellmanFord returning Series."""

    edges = pd.DataFrame(
        data={
            "tail": [0, 0, 1, 2],
            "head": [1, 2, 2, 3],
            "weight": [1.0, 3.0, 1.0, 1.0],
        }
    )

    bf = BellmanFord(edges)
    result = bf.run(vertex_idx=0, return_series=True)

    assert isinstance(result, pd.Series)
    assert len(result) == 4
    assert result[0] == 0.0
    assert result[1] == 1.0
    assert result[2] == 2.0
    assert result[3] == 3.0


def test_bellman_ford_check_edges_allows_negative():
    """Test that BellmanFord allows negative weights with check_edges."""

    edges = pd.DataFrame(
        data={
            "tail": [0, 0],
            "head": [1, 2],
            "weight": [1.0, -2.0],
        }
    )

    # BellmanFord should allow negative weights
    bf = BellmanFord(edges, check_edges=True)
    assert bf is not None

    # Dijkstra should reject negative weights
    with pytest.raises(ValueError, match=r"nonnegative"):
        Dijkstra(edges, check_edges=True)


# ============================================================================ #
# BFS tests                                                                    #
# ============================================================================ #


def test_bfs_scipy_comparison(random_seed=124, n=1000):
    """Test BFS against scipy.sparse.csgraph.breadth_first_order."""
    np.random.seed(random_seed)
    tail = np.random.randint(0, int(n / 5), n)
    head = np.random.randint(0, int(n / 5), n)
    # BFS doesn't use weights, but include for consistency with graph structure
    weight = np.random.rand(n)
    edges = pd.DataFrame(data={"tail": tail, "head": head, "weight": weight})
    edges.drop_duplicates(subset=["tail", "head"], inplace=True)
    edges = edges.loc[edges["tail"] != edges["head"]]
    edges.reset_index(drop=True, inplace=True)

    # SciPy BFS
    vertex_count = edges[["tail", "head"]].max().max() + 1
    # For BFS, we can use unit weights (or no weights)
    data = np.ones(len(edges), dtype=np.float64)
    row = edges["tail"].values.astype(np.int32)
    col = edges["head"].values.astype(np.int32)
    graph_coo = coo_array((data, (row, col)), shape=(vertex_count, vertex_count))
    graph_csr = graph_coo.tocsr()

    # Run scipy's breadth_first_order from vertex 0
    node_array_ref, predecessors_ref = breadth_first_order(
        csgraph=graph_csr, i_start=0, directed=True, return_predecessors=True
    )

    # In-house BFS
    # Test with orientation="out" and permute=False
    bfs = BFS(edges, orientation="out", check_edges=False, permute=False)
    predecessors = bfs.run(vertex_idx=0)

    # Compare predecessors for all reachable nodes
    # scipy's node_array contains the reachable vertices
    for node in node_array_ref:
        assert (
            predecessors[node] == predecessors_ref[node]
        ), f"Predecessor mismatch at node {node}: {predecessors[node]} != {predecessors_ref[node]}"

    # Check that unreachable nodes have sentinel value
    unreachable_nodes = set(range(vertex_count)) - set(node_array_ref)
    for node in unreachable_nodes:
        assert (
            predecessors[node] == bfs.UNREACHABLE
        ), f"Unreachable node {node} should have sentinel value"

    # Test with permute=True
    bfs_permute = BFS(edges, orientation="out", check_edges=False, permute=True)
    predecessors_permute = bfs_permute.run(vertex_idx=0)

    # Compare results with permute=True
    for node in node_array_ref:
        assert (
            predecessors_permute[node] == predecessors_ref[node]
        ), f"Permute: Predecessor mismatch at node {node}"


# ============================================================================ #
# DataFrame Backend Cross-Compatibility Tests                                 #
# ============================================================================ #


class TestDataFrameBackendCompatibility:
    """Test that algorithms work consistently across different DataFrame backends."""

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

    def test_dijkstra_backend_consistency(self):
        """Test Dijkstra consistency across DataFrame backends."""
        data = {
            "tail": [0, 0, 1, 1, 2],
            "head": [1, 2, 2, 3, 3],
            "weight": [1.0, 4.0, 2.0, 5.0, 1.0],
        }

        dataframes = self.create_test_dataframes(data)
        results = {}

        # Run Dijkstra on each DataFrame backend
        for backend_name, df in dataframes.items():
            dijkstra = Dijkstra(df, orientation="out")
            results[backend_name] = dijkstra.run(vertex_idx=0, return_inf=True)

        # All results should be identical
        reference_result = list(results.values())[0]
        for backend_name, result in results.items():
            np.testing.assert_array_equal(
                reference_result,
                result,
                err_msg=f"Dijkstra results differ for {backend_name}",
            )

    def test_bellmanford_backend_consistency(self):
        """Test BellmanFord consistency across DataFrame backends."""
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
                err_msg=f"BellmanFord results differ for {backend_name}",
            )

    def test_hyperpath_backend_consistency(self):
        """Test HyperpathGenerating consistency across DataFrame backends."""
        data = {
            "tail": [0, 0, 1, 2],
            "head": [1, 2, 2, 3],
            "trav_time": [1.0, 2.0, 1.0, 1.0],
            "freq": [0.1, 0.1, 0.1, 0.1],
        }

        dataframes = self.create_test_dataframes(data)
        results = {}

        # Run HyperpathGenerating on each DataFrame backend
        for backend_name, df in dataframes.items():
            hp = HyperpathGenerating(df)
            hp.run(origin=0, destination=3, volume=1.0, return_inf=True)
            results[backend_name] = hp.u_i_vec.copy()

        # All results should be identical (within tolerance)
        reference_result = list(results.values())[0]
        for backend_name, result in results.items():
            np.testing.assert_allclose(
                reference_result,
                result,
                rtol=1e-10,
                err_msg=f"HyperpathGenerating results differ for {backend_name}",
            )

    def test_path_tracking_backend_consistency(self):
        """Test path tracking consistency across DataFrame backends."""
        data = {
            "tail": [0, 0, 1, 1, 2],
            "head": [1, 2, 2, 3, 3],
            "weight": [1.0, 4.0, 2.0, 5.0, 1.0],
        }

        dataframes = self.create_test_dataframes(data)
        paths = {}

        # Run Dijkstra with path tracking on each DataFrame backend
        for backend_name, df in dataframes.items():
            dijkstra = Dijkstra(df, orientation="out")
            dijkstra.run(vertex_idx=0, path_tracking=True, return_inf=True)
            paths[backend_name] = dijkstra.get_path(3)

        # All paths should be identical
        reference_path = list(paths.values())[0]
        for backend_name, path in paths.items():
            if reference_path is None:
                assert path is None, f"Path differs for {backend_name}: expected None"
            else:
                np.testing.assert_array_equal(
                    reference_path,
                    path,
                    err_msg=f"Path tracking results differ for {backend_name}",
                )

    def test_custom_column_names_backend_consistency(self):
        """Test custom column names work consistently across backends."""
        data = {
            "from_node": [0, 0, 1, 2],
            "to_node": [1, 2, 2, 3],
            "cost": [1.0, 4.0, 2.0, 1.0],
        }

        dataframes = self.create_test_dataframes(data)
        results = {}

        # Run with custom column names
        for backend_name, df in dataframes.items():
            dijkstra = Dijkstra(df, tail="from_node", head="to_node", weight="cost")
            results[backend_name] = dijkstra.run(vertex_idx=0, return_inf=True)

        # All results should be identical
        reference_result = list(results.values())[0]
        for backend_name, result in results.items():
            np.testing.assert_array_equal(
                reference_result,
                result,
                err_msg=f"Custom column results differ for {backend_name}",
            )

    def test_internal_dtype_optimization_consistency(self):
        """Test that internal dtype optimization works across backends."""
        data = {
            "tail": [0, 1, 2, 3],
            "head": [1, 2, 3, 4],
            "weight": [1.0, 2.0, 3.0, 4.0],
        }

        dataframes = self.create_test_dataframes(data)

        # Check internal representations are consistent
        for backend_name, df in dataframes.items():
            dijkstra = Dijkstra(df)

            # All backends should use optimal internal dtypes for weights
            assert (
                dijkstra._edges["weight"].dtype == np.float64
            ), f"{backend_name} weight dtype"

            # Memory should be contiguous for all backends
            assert dijkstra._edges["tail"].values.flags[
                "C_CONTIGUOUS"
            ], f"{backend_name} tail not contiguous"
            assert dijkstra._edges["head"].values.flags[
                "C_CONTIGUOUS"
            ], f"{backend_name} head not contiguous"
            assert dijkstra._edges["weight"].values.flags[
                "C_CONTIGUOUS"
            ], f"{backend_name} weight not contiguous"

            # For vertex indices, different backends may use different dtypes but should be integer types
            assert np.issubdtype(
                dijkstra._edges["tail"].dtype, np.integer
            ), f"{backend_name} tail should be integer"
            assert np.issubdtype(
                dijkstra._edges["head"].dtype, np.integer
            ), f"{backend_name} head should be integer"

    def test_permutation_backend_consistency(self):
        """Test vertex permutation works consistently across backends."""
        data = {
            "tail": [10, 10, 20, 30],
            "head": [20, 30, 30, 40],
            "weight": [1.0, 4.0, 2.0, 1.0],
        }

        dataframes = self.create_test_dataframes(data)
        results = {}

        # Run with permute=True
        for backend_name, df in dataframes.items():
            dijkstra = Dijkstra(df, permute=True)
            results[backend_name] = dijkstra.run(vertex_idx=10, return_inf=True)

        # All results should be identical
        reference_result = list(results.values())[0]
        for backend_name, result in results.items():
            np.testing.assert_array_equal(
                reference_result,
                result,
                err_msg=f"Permutation results differ for {backend_name}",
            )

    def test_large_graph_backend_consistency(self):
        """Test consistency with larger graphs across backends."""
        np.random.seed(42)  # For reproducible results
        n = 100
        m = 500

        data = {
            "tail": np.random.randint(0, n, m).tolist(),
            "head": np.random.randint(0, n, m).tolist(),
            "weight": np.random.uniform(0.1, 10.0, m).tolist(),
        }

        dataframes = self.create_test_dataframes(data)
        results = {}

        # Run on larger graphs
        for backend_name, df in dataframes.items():
            dijkstra = Dijkstra(df)
            results[backend_name] = dijkstra.run(vertex_idx=0, return_inf=True)

        # All results should be identical
        reference_result = list(results.values())[0]
        for backend_name, result in results.items():
            np.testing.assert_array_equal(
                reference_result,
                result,
                err_msg=f"Large graph results differ for {backend_name}",
            )
