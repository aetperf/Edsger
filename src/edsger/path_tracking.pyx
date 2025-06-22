"""
Path tracking module.

cpdef functions:

- compute_path
    Compute path from predecessors or successors.

cdef functions:

- _compute_path_first_pass
    Returns the path length.
- _compute_path_second_pass
    Compute the sequence of vertices forming a path.
"""

import numpy as np
cimport numpy as cnp


cpdef cnp.ndarray compute_path(cnp.uint32_t[::1] path_links, int vertex_idx):
    """Compute path from predecessors or successors.

    Parameters:
    -----------

    path_links : cnp.uint32_t[::1]
        predecessors or successors.

    vertex_idx : int
        source or target vertex index.
    """

    cdef int path_length

    path_length = _compute_path_first_pass(path_links, vertex_idx)
    path_vertices = np.empty(path_length, dtype=np.uint32)
    _compute_path_second_pass(path_links, path_vertices, vertex_idx)

    return path_vertices


cdef int _compute_path_first_pass(
    cnp.uint32_t[::1] path_links,
    int vertex_idx
) noexcept nogil:
    """Returns the path length.
    """

    cdef:
        size_t i, j
        int k

    # initialization
    j = <size_t>vertex_idx
    i = j + 1
    k = 0

    # loop
    while i != j:
        i = j
        j = <size_t>path_links[j]
        k += 1

    return k


cdef void _compute_path_second_pass(
    cnp.uint32_t[::1] path_links,
    cnp.uint32_t[::1] path_vertices,
    int vertex_idx
) noexcept nogil:
    """Compute the sequence of vertices forming a path.
    """
    cdef size_t i, j, k

    # initialization
    j = <size_t>vertex_idx
    i = j + 1
    k = 0

    # loop
    while i != j:
        i = j
        path_vertices[k] = j
        j = <size_t>path_links[j]
        k += 1


# author : Francois Pacull
# copyright : Architecture & Performance
# email: francois.pacull@architecture-performance.fr
# license : MIT
