"""Type stubs for edsger.kruskal Cython module."""

import numpy as np
import numpy.typing as npt

def kruskal(
    tails: npt.NDArray[np.uint32],
    heads: npt.NDArray[np.uint32],
    order: npt.NDArray[np.int64],
    vertex_count: int,
) -> npt.NDArray[np.int64]: ...
