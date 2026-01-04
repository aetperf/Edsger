"""Type stubs for edsger.dijkstra Cython module."""

import numpy as np
import numpy.typing as npt

def compute_sssp(
    csr_indptr: npt.NDArray[np.uint32],
    csr_indices: npt.NDArray[np.uint32],
    csr_data: npt.NDArray[np.float64],
    source_vert_idx: int,
    vertex_count: int,
    heap_length: int,
) -> npt.NDArray[np.float64]: ...
def compute_sssp_w_path(
    csr_indptr: npt.NDArray[np.uint32],
    csr_indices: npt.NDArray[np.uint32],
    csr_data: npt.NDArray[np.float64],
    predecessor: npt.NDArray[np.uint32],
    source_vert_idx: int,
    vertex_count: int,
    heap_length: int,
) -> npt.NDArray[np.float64]: ...
def compute_sssp_early_termination(
    csr_indptr: npt.NDArray[np.uint32],
    csr_indices: npt.NDArray[np.uint32],
    csr_data: npt.NDArray[np.float64],
    termination_nodes: npt.NDArray[np.uint32],
    source_vert_idx: int,
    vertex_count: int,
    heap_length: int,
) -> npt.NDArray[np.float64]: ...
def compute_sssp_w_path_early_termination(
    csr_indptr: npt.NDArray[np.uint32],
    csr_indices: npt.NDArray[np.uint32],
    csr_data: npt.NDArray[np.float64],
    predecessor: npt.NDArray[np.uint32],
    termination_nodes: npt.NDArray[np.uint32],
    source_vert_idx: int,
    vertex_count: int,
    heap_length: int,
) -> npt.NDArray[np.float64]: ...
def compute_stsp(
    csc_indptr: npt.NDArray[np.uint32],
    csc_indices: npt.NDArray[np.uint32],
    csc_data: npt.NDArray[np.float64],
    target_vert_idx: int,
    vertex_count: int,
    heap_length: int,
) -> npt.NDArray[np.float64]: ...
def compute_stsp_w_path(
    csc_indptr: npt.NDArray[np.uint32],
    csc_indices: npt.NDArray[np.uint32],
    csc_data: npt.NDArray[np.float64],
    successor: npt.NDArray[np.uint32],
    target_vert_idx: int,
    vertex_count: int,
    heap_length: int,
) -> npt.NDArray[np.float64]: ...
def compute_stsp_early_termination(
    csc_indptr: npt.NDArray[np.uint32],
    csc_indices: npt.NDArray[np.uint32],
    csc_data: npt.NDArray[np.float64],
    termination_nodes: npt.NDArray[np.uint32],
    target_vert_idx: int,
    vertex_count: int,
    heap_length: int,
) -> npt.NDArray[np.float64]: ...
def compute_stsp_w_path_early_termination(
    csc_indptr: npt.NDArray[np.uint32],
    csc_indices: npt.NDArray[np.uint32],
    csc_data: npt.NDArray[np.float64],
    successor: npt.NDArray[np.uint32],
    termination_nodes: npt.NDArray[np.uint32],
    target_vert_idx: int,
    vertex_count: int,
    heap_length: int,
) -> npt.NDArray[np.float64]: ...
def compute_sssp_01() -> None: ...
def compute_stsp_01() -> None: ...
def compute_sssp_02() -> None: ...
def compute_stsp_02() -> None: ...
def compute_sssp_early_termination_01() -> None: ...
def compute_stsp_early_termination_01() -> None: ...
