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

# Spiess & Florian
# ----------------

# infinite frequency is defined here numerically
# this must be a very large number depending on the precision on the computation
# INF_FREQ << DTYPE_INF
cdef DTYPE_t INF_FREQ = 1.0e+20
INF_FREQ_PY = INF_FREQ

# smallest frequency
# WARNING: this must be small but not too small 
# 1 / MIN_FREQ << DTYPE_INF
cdef DTYPE_t MIN_FREQ = 1.0e-20
MIN_FREQ_PY =  MIN_FREQ

# a very small time interval
cdef DTYPE_t A_VERY_SMALL_TIME_INTERVAL = 1.0e-12
A_VERY_SMALL_TIME_INTERVAL_PY = A_VERY_SMALL_TIME_INTERVAL
