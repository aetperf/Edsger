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


# argument checking
# -----------------

cdef int check_graph_arguments(
    cnp.uint32_t[::1] indptr,
    cnp.uint32_t[::1] indices,
    Py_ssize_t data_length,
    int vert_idx,
    int vertex_count) except -1


# author : Francois Pacull
# copyright : Architecture & Performance
# email: francois.pacull@architecture-performance.fr
# license : MIT
