"""Type stubs for edsger.path_tracking Cython module."""

import numpy as np
import numpy.typing as npt

def compute_path(
    path_links: npt.NDArray[np.uint32],
    vertex_idx: int,
) -> npt.NDArray[np.uint32]: ...
