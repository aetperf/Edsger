"""Type stubs for edsger.components_core Cython module."""

import numpy as np
import numpy.typing as npt

def weak_components_csr(
    csr_indptr: npt.NDArray[np.uint32],
    csr_indices: npt.NDArray[np.uint32],
    vertex_count: int,
) -> tuple[npt.NDArray[np.uint32], int]: ...
def strong_components_csr(
    csr_indptr: npt.NDArray[np.uint32],
    csr_indices: npt.NDArray[np.uint32],
    vertex_count: int,
) -> tuple[npt.NDArray[np.uint32], int]: ...
def test_components_01() -> None: ...
