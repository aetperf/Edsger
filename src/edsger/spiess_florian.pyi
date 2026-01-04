"""Type stubs for edsger.spiess_florian Cython module."""

import numpy as np
import numpy.typing as npt

def compute_SF_in(
    csc_indptr: npt.NDArray[np.uint32],
    csc_edge_idx: npt.NDArray[np.uint32],
    c_a_vec: npt.NDArray[np.float64],
    f_a_vec: npt.NDArray[np.float64],
    tail_indices: npt.NDArray[np.uint32],
    head_indices: npt.NDArray[np.uint32],
    demand_indices: npt.NDArray[np.uint32],
    demand_values: npt.NDArray[np.float64],
    v_a_vec: npt.NDArray[np.float64],
    u_i_vec: npt.NDArray[np.float64],
    vertex_count: int,
    dest_vert_index: int,
) -> None: ...
def compute_SF_in_01() -> None: ...
def compute_SF_in_02() -> None: ...
