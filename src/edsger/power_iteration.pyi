"""Type stubs for edsger.power_iteration Cython module."""

import numpy as np
import numpy.typing as npt

def pagerank_csc(
    csc_indptr: npt.NDArray[np.uint32],
    csc_indices: npt.NDArray[np.uint32],
    csc_data: npt.NDArray[np.float64],
    weighted: bool,
    inv_out_weight: npt.NDArray[np.float64],
    teleport: npt.NDArray[np.float64],
    dangling_weights: npt.NDArray[np.float64],
    x0: npt.NDArray[np.float64],
    damping: float,
    tol: float,
    max_iter: int,
) -> tuple[npt.NDArray[np.float64], int]: ...
def hits_csr_csc(
    csr_indptr: npt.NDArray[np.uint32],
    csr_indices: npt.NDArray[np.uint32],
    csr_data: npt.NDArray[np.float64],
    csc_indptr: npt.NDArray[np.uint32],
    csc_indices: npt.NDArray[np.uint32],
    csc_data: npt.NDArray[np.float64],
    weighted: bool,
    h0: npt.NDArray[np.float64],
    tol: float,
    max_iter: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], int]: ...
def katz_csc(
    csc_indptr: npt.NDArray[np.uint32],
    csc_indices: npt.NDArray[np.uint32],
    csc_data: npt.NDArray[np.float64],
    weighted: bool,
    beta: npt.NDArray[np.float64],
    alpha: float,
    tol: float,
    max_iter: int,
) -> tuple[npt.NDArray[np.float64], int]: ...
