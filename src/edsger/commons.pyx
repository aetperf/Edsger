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

    Note that the content of the arrays is not checked: this is the purpose
    of validate_graph, which is not called here because of its linear cost.
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


def validate_graph(
        cnp.uint32_t[::1] indptr,
        cnp.uint32_t[::1] indices,
        data=None,
        bint nonnegative=True):
    """
    Check that a graph in CSR or CSC format is valid.

    The graph algorithms of edsger read these arrays without bounds checking
    and only check, in constant time, that their arguments are consistent.
    This function checks the content of the arrays, in linear time. It is
    meant to be called once per graph, when the arrays do not come from
    edsger.path, before running the algorithms as many times as needed.

    Parameters
    ----------
    indptr : cnp.uint32_t[::1]
        Pointers in the CSR/CSC format: vertex_count + 1 elements.
    indices : cnp.uint32_t[::1]
        Indices in the CSR/CSC format: one element per edge.
    data : array-like, optional
        Edge weights in the CSR/CSC format: one element per edge.
    nonnegative : bool, optional (default=True)
        Whether the edge weights should be non-negative, which is required by
        Dijkstra's algorithm but not by the Bellman-Ford algorithm.

    Raises
    ------
    ValueError
        If the pointers do not start from zero, are not non-decreasing or do
        not end with the edge count, if an index is not smaller than the
        vertex count, or if the weights have the wrong length, contain a NaN
        or, when nonnegative is True, a negative value.
    """
    cdef:
        Py_ssize_t vertex_count = indptr.shape[0] - 1
        Py_ssize_t edge_count = indices.shape[0]
        Py_ssize_t i
        bint unordered = False, out_of_bounds = False

    if vertex_count < 1:
        raise ValueError("the pointer array should have at least 2 elements")
    if indptr[0] != 0:
        raise ValueError("the first pointer should be 0")
    if indptr[vertex_count] != edge_count:
        raise ValueError(
            f"the last pointer should be the edge count {edge_count}, got "
            f"{indptr[vertex_count]}"
        )

    with nogil:
        for i in range(vertex_count):
            unordered |= indptr[i] > indptr[i + 1]
        for i in range(edge_count):
            out_of_bounds |= indices[i] >= vertex_count

    if unordered:
        raise ValueError("the pointers should be non-decreasing")
    if out_of_bounds:
        raise ValueError(
            f"the indices should be smaller than the vertex count {vertex_count}"
        )

    if data is not None:
        weights = np.asarray(data)
        if weights.shape != (edge_count,):
            raise ValueError(
                f"the edge weights should have one element per edge: {edge_count}"
            )
        if edge_count > 0:
            # the min is NaN if any weight is; it is compared as a Python
            # object, away from the fast-math flags of this module
            weight_min = weights.min()
            if weight_min != weight_min:
                raise ValueError("the edge weights should not contain NaN values")
            if nonnegative and weight_min < 0.0:
                raise ValueError("the edge weights should be nonnegative")


# author : Francois Pacull
# copyright : Architecture & Performance
# email: francois.pacull@architecture-performance.fr
# license : MIT
