"""Type stubs for edsger.dfs Cython module."""

import numpy as np
import numpy.typing as npt

def dfs_csr(
    csr_indptr: npt.NDArray[np.uint32],
    csr_indices: npt.NDArray[np.uint32],
    start_vert_idx: int,
    vertex_count: int,
    sentinel: int = -9999,
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.uint32]]: ...
def dfs_csc(
    csc_indptr: npt.NDArray[np.uint32],
    csc_indices: npt.NDArray[np.uint32],
    start_vert_idx: int,
    vertex_count: int,
    sentinel: int = -9999,
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.uint32]]: ...
def test_dfs_csr_01() -> None: ...
def test_dfs_csc_01() -> None: ...
