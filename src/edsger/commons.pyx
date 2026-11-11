# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: embedsignature=False
# cython: cdivision=True
# cython: initializedcheck=False

"""
Common definitions.
"""

cimport numpy as cnp
import numpy as np

DTYPE_PY = np.float64
DTYPE_INF = <DTYPE_t>np.finfo(dtype=DTYPE_PY).max
DTYPE_INF_PY = DTYPE_INF

# Spiess & Florian
# ----------------

# infinite frequency is defined here numerically
# this must be a very large number depending on the precision on the computation
# INF_FREQ << DTYPE_INF
INF_FREQ = 1.0e+20
INF_FREQ_PY = INF_FREQ

# smallest frequency
# WARNING: this must be small but not too small
# 1 / MIN_FREQ << DTYPE_INF
MIN_FREQ = 1.0 / INF_FREQ
MIN_FREQ_PY = MIN_FREQ

# a very small time interval
A_VERY_SMALL_TIME_INTERVAL = 1.0e+08 * MIN_FREQ
A_VERY_SMALL_TIME_INTERVAL_PY = A_VERY_SMALL_TIME_INTERVAL


# argument checking
# -----------------

cdef int check_graph_arguments(
        cnp.uint32_t[::1] indptr,
        cnp.uint32_t[::1] indices,
        Py_ssize_t data_length,
        int vert_idx,
        int vertex_count) except -1:
    """
    Check the arguments of a graph algorithm working on a graph in CSR or CSC
    format. Bounds checking is disabled in the Cython modules, so the
    functions that can be called from Python must check that their arguments
    are consistent before using them as array indices.

    input
    =====
    * cnp.uint32_t[::1] indptr : pointers in the CSR/CSC format
    * cnp.uint32_t[::1] indices : indices in the CSR/CSC format
    * Py_ssize_t data_length : length of the edge data array, or a negative
      value if there is none
    * int vert_idx : origin vertex index
    * int vertex_count : vertex count

    output
    ======
    * int : 0, or -1 with a ValueError (inconsistent graph arrays) or an
      IndexError (vertex index out of bounds) set

    Note that the content of indices is not checked.
    """
    if vertex_count < 1 or indptr.shape[0] != vertex_count + 1:
        raise ValueError(
            f"the pointer array should have vertex_count + 1 = {vertex_count + 1} "
            f"elements, got {indptr.shape[0]}"
        )
    if indptr[vertex_count] > indices.shape[0]:
        raise ValueError(
            f"the pointer array refers to {indptr[vertex_count]} edges, the index "
            f"array has {indices.shape[0]} elements"
        )
    if data_length >= 0 and data_length != indices.shape[0]:
        raise ValueError(
            f"the edge data array has {data_length} elements, the index array has "
            f"{indices.shape[0]} elements"
        )
    if vert_idx < 0 or vert_idx >= vertex_count:
        raise IndexError(
            f"vertex index {vert_idx} is out of bounds for {vertex_count} vertices"
        )
    return 0


# author : Francois Pacull
# copyright : Architecture & Performance
# email: francois.pacull@architecture-performance.fr
# license : MIT
