"""Type stubs for edsger.bellman_ford Cython module."""

import numpy as np
import numpy.typing as npt

def compute_bf_sssp(
    csr_indptr: npt.NDArray[np.uint32],
    csr_indices: npt.NDArray[np.uint32],
    csr_data: npt.NDArray[np.float64],
    source_vert_idx: int,
    vertex_count: int,
) -> npt.NDArray[np.float64]: ...
def compute_bf_sssp_w_path(
    csr_indptr: npt.NDArray[np.uint32],
    csr_indices: npt.NDArray[np.uint32],
    csr_data: npt.NDArray[np.float64],
    predecessor: npt.NDArray[np.uint32],
    source_vert_idx: int,
    vertex_count: int,
) -> npt.NDArray[np.float64]: ...
def compute_bf_stsp(
    csc_indptr: npt.NDArray[np.uint32],
    csc_indices: npt.NDArray[np.uint32],
    csc_data: npt.NDArray[np.float64],
    target_vert_idx: int,
    vertex_count: int,
) -> npt.NDArray[np.float64]: ...
def compute_bf_stsp_w_path(
    csc_indptr: npt.NDArray[np.uint32],
    csc_indices: npt.NDArray[np.uint32],
    csc_data: npt.NDArray[np.float64],
    successor: npt.NDArray[np.uint32],
    target_vert_idx: int,
    vertex_count: int,
) -> npt.NDArray[np.float64]: ...
def detect_negative_cycle(
    csr_indptr: npt.NDArray[np.uint32],
    csr_indices: npt.NDArray[np.uint32],
    csr_data: npt.NDArray[np.float64],
    dist_matrix: npt.NDArray[np.float64],
    vertex_count: int,
) -> bool: ...
def detect_negative_cycle_csc(
    csc_indptr: npt.NDArray[np.uint32],
    csc_indices: npt.NDArray[np.uint32],
    csc_data: npt.NDArray[np.float64],
    stsp_dist: npt.NDArray[np.float64],
    vertex_count: int,
) -> bool: ...
def test_bf_negative_edges() -> None: ...
def test_bf_negative_cycle() -> None: ...
