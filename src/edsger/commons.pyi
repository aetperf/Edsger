"""Type stubs for edsger.commons Cython module."""

from typing import Optional, Type
import numpy as np
import numpy.typing as npt

DTYPE_PY: Type[np.float64]
DTYPE_INF_PY: float
INF_FREQ_PY: float
MIN_FREQ_PY: float
A_VERY_SMALL_TIME_INTERVAL_PY: float

def validate_graph(
    indptr: npt.NDArray[np.uint32],
    indices: npt.NDArray[np.uint32],
    data: Optional[npt.ArrayLike] = None,
    nonnegative: bool = True,
) -> None: ...
