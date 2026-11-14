"""Type stubs for edsger.astar Cython module."""

import numpy as np
import numpy.typing as npt

EARTH_RADIUS: float
METRICS: tuple[str, ...]

class Workspace:
    vertex_count: int
    def __init__(self, vertex_count: int) -> None: ...
    def get_predecessor(self) -> npt.NDArray[np.uint32] | None: ...

def compute_astar(
    csr_indptr: npt.NDArray[np.uint32],
    csr_indices: npt.NDArray[np.uint32],
    csr_data: npt.NDArray[np.float64],
    source_vert_idx: int,
    target_vert_idx: int,
    vertex_count: int,
    heuristic: npt.NDArray[np.float64] | None = None,
    path_tracking: bool = False,
    workspace: Workspace | None = None,
    heap_length: int = 1024,
) -> float: ...
def compute_astar_coordinates(
    csr_indptr: npt.NDArray[np.uint32],
    csr_indices: npt.NDArray[np.uint32],
    csr_data: npt.NDArray[np.float64],
    source_vert_idx: int,
    target_vert_idx: int,
    vertex_count: int,
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    metric: str,
    factor: float,
    path_tracking: bool = False,
    workspace: Workspace | None = None,
    heap_length: int = 1024,
) -> float: ...
