"""Tests for the input validation of the Cython entry points.

Bounds checking is disabled in the Cython modules, so the functions that can
be called from Python check their arguments explicitly.

py.test tests/test_input_validation.py
"""

import numpy as np
import pytest

from edsger import dijkstra, dijkstra_nodec
from edsger.bellman_ford import (
    compute_bf_sssp,
    compute_bf_sssp_w_path,
    compute_bf_stsp,
    compute_bf_stsp_w_path,
    detect_negative_cycle,
    detect_negative_cycle_csc,
)
from edsger.bfs import bfs_csc, bfs_csr, bfs_reorder_csr
from edsger.path_tracking import compute_path
from edsger.spiess_florian import compute_SF_in  # type: ignore

VERTEX_COUNT = 2


def single_edge_graph():
    """Graph with 2 vertices and the single edge 0 -> 1, in CSR format."""
    indptr = np.array([0, 1, 1], dtype=np.uint32)
    indices = np.array([1], dtype=np.uint32)
    data = np.array([1.0], dtype=np.float64)
    return indptr, indices, data


def run_dijkstra(module, name, vertex_idx, vertex_count=VERTEX_COUNT):
    indptr, indices, data = single_edge_graph()
    return getattr(module, name)(
        indptr, indices, data, vertex_idx, vertex_count, vertex_count
    )


def run_dijkstra_w_path(module, name, vertex_idx, vertex_count=VERTEX_COUNT):
    indptr, indices, data = single_edge_graph()
    path = np.arange(VERTEX_COUNT, dtype=np.uint32)
    return getattr(module, name)(
        indptr, indices, data, path, vertex_idx, vertex_count, vertex_count
    )


def run_bf(func, vertex_idx, vertex_count=VERTEX_COUNT):
    indptr, indices, data = single_edge_graph()
    return func(indptr, indices, data, vertex_idx, vertex_count)


def run_bf_w_path(func, vertex_idx, vertex_count=VERTEX_COUNT):
    indptr, indices, data = single_edge_graph()
    path = np.arange(VERTEX_COUNT, dtype=np.uint32)
    return func(indptr, indices, data, path, vertex_idx, vertex_count)


def run_bfs(func, vertex_idx, vertex_count=VERTEX_COUNT):
    indptr, indices, _ = single_edge_graph()
    return func(indptr, indices, vertex_idx, vertex_count)


ENTRY_POINTS = [
    pytest.param(
        lambda v, n: run_dijkstra(dijkstra, "compute_sssp", v, n), id="d-sssp"
    ),
    pytest.param(
        lambda v, n: run_dijkstra(dijkstra, "compute_stsp", v, n), id="d-stsp"
    ),
    pytest.param(
        lambda v, n: run_dijkstra_w_path(dijkstra, "compute_sssp_w_path", v, n),
        id="d-sssp-path",
    ),
    pytest.param(
        lambda v, n: run_dijkstra(dijkstra_nodec, "compute_sssp", v, n), id="lazy-sssp"
    ),
    pytest.param(
        lambda v, n: run_dijkstra(dijkstra_nodec, "compute_stsp", v, n), id="lazy-stsp"
    ),
    pytest.param(
        lambda v, n: run_dijkstra_w_path(dijkstra_nodec, "compute_stsp_w_path", v, n),
        id="lazy-stsp-path",
    ),
    pytest.param(lambda v, n: run_bf(compute_bf_sssp, v, n), id="bf-sssp"),
    pytest.param(lambda v, n: run_bf(compute_bf_stsp, v, n), id="bf-stsp"),
    pytest.param(
        lambda v, n: run_bf_w_path(compute_bf_sssp_w_path, v, n), id="bf-sssp-path"
    ),
    pytest.param(
        lambda v, n: run_bf_w_path(compute_bf_stsp_w_path, v, n), id="bf-stsp-path"
    ),
    pytest.param(lambda v, n: run_bfs(bfs_csr, v, n), id="bfs-csr"),
    pytest.param(lambda v, n: run_bfs(bfs_csc, v, n), id="bfs-csc"),
    pytest.param(lambda v, n: run_bfs(bfs_reorder_csr, v, n), id="bfs-reorder"),
]


@pytest.mark.parametrize("run", ENTRY_POINTS)
@pytest.mark.parametrize("vertex_idx", [-1, VERTEX_COUNT, 50_000_000])
def test_origin_vertex_out_of_bounds(run, vertex_idx):
    with pytest.raises(IndexError):
        run(vertex_idx, VERTEX_COUNT)


@pytest.mark.parametrize("run", ENTRY_POINTS)
@pytest.mark.parametrize("vertex_count", [0, 1, 3, 1000])
def test_vertex_count_not_matching_indptr(run, vertex_count):
    """indptr has 3 elements: the only valid vertex count is 2."""
    with pytest.raises(ValueError):
        run(0, vertex_count)


@pytest.mark.parametrize("module", [dijkstra, dijkstra_nodec])
def test_dijkstra_termination_node_out_of_bounds(module):
    indptr, indices, data = single_edge_graph()
    termination_nodes = np.array([1, 7], dtype=np.uint32)
    with pytest.raises(IndexError):
        module.compute_sssp_early_termination(
            indptr, indices, data, termination_nodes, 0, VERTEX_COUNT, VERTEX_COUNT
        )


@pytest.mark.parametrize("module", [dijkstra, dijkstra_nodec])
def test_dijkstra_path_array_too_short(module):
    indptr, indices, data = single_edge_graph()
    predecessor = np.arange(1, dtype=np.uint32)
    with pytest.raises(ValueError):
        module.compute_sssp_w_path(
            indptr, indices, data, predecessor, 0, VERTEX_COUNT, VERTEX_COUNT
        )


def test_lazy_dijkstra_terminates_on_negative_cycle():
    """Without a vertex state, the lazy algorithm would loop forever on a
    negative cycle: 0 -> 1 -> 2 -> 1 with weights 1, -2, 1."""
    indptr = np.array([0, 1, 2, 3], dtype=np.uint32)
    indices = np.array([1, 2, 1], dtype=np.uint32)
    data = np.array([1.0, -2.0, 1.0], dtype=np.float64)
    with pytest.raises(ValueError, match="negative"):
        dijkstra_nodec.compute_sssp(indptr, indices, data, 0, 3, 3)


@pytest.mark.parametrize("func", [detect_negative_cycle, detect_negative_cycle_csc])
def test_detect_negative_cycle_dist_array_too_short(func):
    indptr, indices, data = single_edge_graph()
    dist = np.zeros(1, dtype=np.float64)
    with pytest.raises(ValueError):
        func(indptr, indices, data, dist, VERTEX_COUNT)


def test_compute_path_link_out_of_bounds():
    """Vertex 1 is linked to vertex 7, which does not exist."""
    path_links = np.array([0, 7, 1], dtype=np.uint32)
    with pytest.raises(IndexError):
        compute_path(path_links, 2)


def spiess_florian_args():
    """Arguments of compute_SF_in for the single edge 0 -> 1, towards vertex 1."""
    return {
        "csc_indptr": np.array([0, 0, 1], dtype=np.uint32),
        "csc_edge_idx": np.array([0], dtype=np.uint32),
        "c_a_vec": np.array([1.0]),
        "f_a_vec": np.array([1.0]),
        "tail_indices": np.array([0], dtype=np.uint32),
        "head_indices": np.array([1], dtype=np.uint32),
        "demand_indices": np.array([0], dtype=np.uint32),
        "demand_values": np.array([1.0]),
        "v_a_vec": np.zeros(1),
        "u_i_vec": np.full(2, np.inf),
        "vertex_count": 2,
        "dest_vert_index": 1,
    }


def test_compute_SF_in_single_edge():
    args = spiess_florian_args()
    compute_SF_in(*args.values())
    np.testing.assert_array_equal(args["v_a_vec"], [1.0])


@pytest.mark.parametrize(
    "name, value",
    [
        ("demand_values", np.array([])),
        ("u_i_vec", np.full(1, np.inf)),
        ("v_a_vec", np.zeros(0)),
        ("c_a_vec", np.zeros(0)),
        ("f_a_vec", np.zeros(0)),
        ("csc_indptr", np.array([0, 1], dtype=np.uint32)),
        ("csc_edge_idx", np.array([5], dtype=np.uint32)),
        ("tail_indices", np.array([9], dtype=np.uint32)),
        ("head_indices", np.array([9], dtype=np.uint32)),
    ],
)
def test_compute_SF_in_inconsistent_arrays(name, value):
    args = spiess_florian_args()
    args[name] = value
    with pytest.raises((ValueError, IndexError)):
        compute_SF_in(*args.values())
