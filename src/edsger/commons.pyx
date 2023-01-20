""" 
Common definitions.

author : Francois Pacull
copyright : Architecture & Performance
email: francois.pacull@architecture-performance.fr
license : MIT
"""

import numpy as np

DTYPE_PY = np.float64
DTYPE_INF = <DTYPE_t>np.finfo(dtype=DTYPE_PY).max
DTYPE_INF_PY = DTYPE_INF