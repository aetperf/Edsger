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
cdef DTYPE_t INF_FREQ = 4.5e+13

# smallest frequency (1/s)
# WARNING: this must be small but not too small 
# we should have
# 1/SMALLFREQUENCY << INFINITETIME
cdef DTYPE_t MIN_FREQ = 1.0e-08

# infinite frequency is defined here numerically
# this must be a very large number depending on the precision on the computation
INF_FREQ_PY = INF_FREQ

# smallest frequency (veh/s)
MIN_FREQ_PY =  MIN_FREQ