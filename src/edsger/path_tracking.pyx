""" 
Path tracking module.
    
author : Francois Pacull
copyright : Architecture & Performance
email: francois.pacull@architecture-performance.fr
license : MIT
"""

import numpy as np
cimport numpy as cnp


cpdef cnp.ndarray  compute_path(cnp.uint32_t[::1] path_links, int vertex_idx):
    
    cdef int path_length

    path_length = compute_path_first_pass(path_links, vertex_idx)
    path_vertices = np.empty(path_length, dtype=np.uint32)
    compute_path_second_pass(path_links, path_vertices, vertex_idx)

    return path_vertices
    

cdef int compute_path_first_pass(
    cnp.uint32_t[::1] path_links,  
    int vertex_idx
) nogil:
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

cdef void compute_path_second_pass(
    cnp.uint32_t[::1] path_links,
    cnp.uint32_t[::1] path_vertices,  
    int vertex_idx
) nogil:

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