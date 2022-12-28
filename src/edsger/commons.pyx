""" Common definitions.

   header file

author : Francois Pacull
copyright : Architecture & Performance
email: francois.pacull@architecture-performance.fr
license : MIT
"""

import numpy as np

DTYPE = np.float64
DTYPE_PY = DTYPE
DTYPE_INF = <DTYPE_t>np.finfo(dtype=DTYPE).max
DTYPE_INF_PY = DTYPE_INF
