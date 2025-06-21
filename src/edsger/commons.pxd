"""
Common definitions. Header file.
"""

cimport numpy as cnp

# priority queue
# --------------

ctypedef cnp.float64_t DTYPE_t
cdef DTYPE_t DTYPE_INF
cdef DTYPE_t INF_FREQ
cdef DTYPE_t MIN_FREQ
cdef DTYPE_t A_VERY_SMALL_TIME_INTERVAL

cdef enum ElementState:
    SCANNED
    UNLABELED
    LABELED


# author : Francois Pacull
# copyright : Architecture & Performance
# email: francois.pacull@architecture-performance.fr
# license : MIT
