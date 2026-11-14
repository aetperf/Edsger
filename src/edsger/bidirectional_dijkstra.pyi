"""Type stubs for edsger.bidirectional_dijkstra Cython module."""

import numpy as np
import numpy.typing as npt

class Workspace:
    vertex_count: int
    def __init__(self, vertex_count: int) -> None: ...
    def get_path_links(
        self,
    ) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]] | None: ...

def compute_bidirectional(
    csr_indptr: npt.NDArray[np.uint32],
    csr_indices: npt.NDArray[np.uint32],
    csr_data: npt.NDArray[np.float64],
    csc_indptr: npt.NDArray[np.uint32],
    csc_indices: npt.NDArray[np.uint32],
    csc_data: npt.NDArray[np.float64],
    source_vert_idx: int,
    target_vert_idx: int,
    vertex_count: int,
    path_tracking: bool = False,
    workspace: Workspace | None = None,
    heap_length: int = 1024,
) -> tuple[float, int]: ...
