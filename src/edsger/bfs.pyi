"""Type stubs for edsger.bfs Cython module."""

import numpy as np
import numpy.typing as npt

def bfs_csr(
    csr_indptr: npt.NDArray[np.uint32],
    csr_indices: npt.NDArray[np.uint32],
    start_vert_idx: int,
    vertex_count: int,
    sentinel: int = -9999,
) -> npt.NDArray[np.int32]: ...
def bfs_csc(
    csc_indptr: npt.NDArray[np.uint32],
    csc_indices: npt.NDArray[np.uint32],
    start_vert_idx: int,
    vertex_count: int,
    sentinel: int = -9999,
) -> npt.NDArray[np.int32]: ...
def test_bfs_csr_01() -> None: ...
def test_bfs_csc_01() -> None: ...
def test_bfs_unreachable() -> None: ...
