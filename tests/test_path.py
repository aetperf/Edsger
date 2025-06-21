"""Tests for path.py.

py.test tests/test_path.py

author : Francois Pacull
copyright : Architecture & Performance
email: francois.pacull@architecture-performance.fr
license : MIT
"""

import numpy as np
import pandas as pd
import pytest
from edsger.commons import A_VERY_SMALL_TIME_INTERVAL_PY, DTYPE_INF_PY
from edsger.networks import create_sf_network
from edsger.path import Dijkstra, HyperpathGenerating
from scipy.sparse import coo_array, csr_matrix
from scipy.sparse.csgraph import dijkstra


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
        sp = Dijkstra(
            edges,
            check_edges=True,
        )


def test_check_edges_02(braess):
    edges = braess

    with pytest.raises(TypeError, match=r"pandas DataFrame"):
        sp = Dijkstra("yeaaahhh!!!", check_edges=True)
    with pytest.raises(KeyError, match=r"not found in graph edges dataframe"):
        sp = Dijkstra(edges, tail="source", check_edges=True)
    with pytest.raises(KeyError, match=r"not found in graph edges dataframe"):
        sp = Dijkstra(edges, head="target", check_edges=True)
    with pytest.raises(KeyError, match=r"not found in graph edges dataframe"):
        sp = Dijkstra(edges, weight="cost", check_edges=True)
    with pytest.raises(ValueError, match=r"missing value"):
        sp = Dijkstra(edges.replace(0, np.nan), check_edges=True)
    with pytest.raises(TypeError, match=r"should be of integer type"):
        sp = Dijkstra(edges.astype({"tail": float}), check_edges=True)
    with pytest.raises(TypeError, match=r"should be of numeric type"):
        sp = Dijkstra(edges.astype({"weight": str}), check_edges=True)


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
    path_lengths = sp.run(
        vertex_idx=0, path_tracking=True, return_inf=True, return_series=False
    )
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

    assert np.allclose(edges["volume_ref"].values, hp._edges["volume"].values)

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
    assert np.allclose(u_i_vec_ref, hp.u_i_vec, rtol=1e-08, atol=1e-08)


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

    assert np.allclose(edges["volume_ref"].values, hp._edges["volume"].values)

    u_i_vec_ref = np.array(
        [
            dwell_time,
            0.5 * (1.0 / line_freq + dwell_time),
            0.0,
        ]
    )
    assert np.allclose(u_i_vec_ref, hp.u_i_vec, rtol=1e-08, atol=1e-08)

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
    assert np.allclose(edges["volume_ref"].values, hp._edges["volume"].values)

    u_i_vec_ref = [305.0, 302.5, 0.0]
    assert np.allclose(u_i_vec_ref, hp.u_i_vec, rtol=1e-08, atol=1e-08)
